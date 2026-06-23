// PFlash prompt compression — scorer-model-driven prefill/KV compression.
// Provenance: ported from buun (github.com/spiritbuun/buun-llama-cpp,
//   remote `buun`, branch experiment/SD-089-pflash). See docs/features/pflash.md.
#include "pflash.h"
#include "pflash-loader.h"
#include "pflash-graph.h"
#include "pflash-score.h"
#include "common.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <cinttypes>
#include <list>
#include <mutex>
#include <unordered_map>
#include <thread>
#include <condition_variable>
#include <functional>
#include <string>
#include <sys/stat.h>

// FNV-1a 64-bit — no external dep, good distribution for short byte strings
static uint64_t fnv1a(const void * data, size_t len, uint64_t h = 14695981039346656037ULL) {
	const auto * p = static_cast<const uint8_t *>(data);
	for (size_t i = 0; i < len; ++i) {
		h ^= p[i];
		h *= 1099511628211ULL;
	}
	return h;
}

// LRU cache for scorer results keyed on (prefix_hash, params_hash)
struct PFlashCacheKey {
	uint64_t prefix_hash;
	uint64_t params_hash;
	bool operator==(const PFlashCacheKey & o) const {
		return prefix_hash == o.prefix_hash && params_hash == o.params_hash;
	}
};

struct PFlashCacheKeyHash {
	size_t operator()(const PFlashCacheKey & k) const {
		// golden-ratio mixing to reduce clustering
		return static_cast<size_t>(k.prefix_hash ^ (k.params_hash * 0x9e3779b97f4a7c15ULL));
	}
};

struct PFlashCacheEntry {
	PFlashCacheKey       key;
	pflash_scorer_result result;
};

// ---------------------------------------------------------------------------
// Disk cache
//
// File format (little-endian binary):
//
//   Header:
//     [8]  magic         = "PFLSHCCH" (no null terminator)
//     [4]  version       = 1  (uint32_t)
//     [8]  scorer_path_hash  (uint64_t — FNV-1a of scorer model path)
//     [8]  scorer_mtime      (int64_t  — st_mtime seconds)
//     [4]  n_records         (uint32_t — number of records that follow)
//
//   Per record (variable length):
//     [8]  prefix_hash  (uint64_t)
//     [8]  params_hash  (uint64_t)
//     [4]  n_lookahead  (int32_t)
//     [4]  seq_len      (int32_t)
//     [4]  n_floats     (uint32_t)   — must equal n_lookahead * seq_len
//     [n_floats * 4]  running_max floats
//
// On any parse error the file is discarded (no crash).
// ---------------------------------------------------------------------------

static constexpr uint8_t PFLASH_CACHE_MAGIC[8] = {'P','F','L','S','H','C','C','H'};
static constexpr uint32_t PFLASH_CACHE_VERSION  = 1;

// Derive the cache file path from scorer path and optional cache_dir.
static std::string pflash_cache_file_path(const std::string & scorer_path,
                                           const std::string & cache_dir) {
	// Extract filename stem from scorer_path
	size_t sep = scorer_path.rfind('/');
	std::string fname = (sep == std::string::npos) ? scorer_path : scorer_path.substr(sep + 1);
	// Append .pflcache extension
	fname += ".pflcache";

	std::string dir;
	if (!cache_dir.empty()) {
		dir = cache_dir;
	} else {
		// Default: same directory as the scorer model
		if (sep != std::string::npos) {
			dir = scorer_path.substr(0, sep);
		} else {
			dir = ".";
		}
	}
	return dir + "/" + fname;
}

// Get scorer model mtime (seconds); returns -1 if stat fails.
static int64_t scorer_mtime(const std::string & path) {
	struct stat st;
	if (stat(path.c_str(), &st) != 0) return -1;
	return (int64_t)st.st_mtime;
}

// Write helpers (little-endian on all common platforms; we use memcpy).
template<typename T>
static void write_pod(FILE * f, T v) {
	fwrite(&v, sizeof(v), 1, f);
}
template<typename T>
static bool read_pod(FILE * f, T & v) {
	return fread(&v, sizeof(v), 1, f) == 1;
}

struct PFlashDiskCacheRecord {
	PFlashCacheKey       key;
	pflash_scorer_result result;
};

// Load all valid records from the cache file. Returns false if the file is
// absent, corrupt, or version/header mismatch (graceful degradation).
static bool pflash_disk_cache_load(
		const std::string & path,
		uint64_t            expected_path_hash,
		int64_t             expected_mtime,
		size_t              lru_cap,
		std::vector<PFlashDiskCacheRecord> & out) {

	FILE * f = fopen(path.c_str(), "rb");
	if (!f) return false; // no cache file — not an error

	bool ok = false;
	do {
		// Read and check magic
		uint8_t magic[8];
		if (fread(magic, 1, 8, f) != 8) break;
		if (memcmp(magic, PFLASH_CACHE_MAGIC, 8) != 0) {
			fprintf(stderr, "pflash: disk cache magic mismatch, ignoring '%s'\n", path.c_str());
			break;
		}

		uint32_t version;
		if (!read_pod(f, version) || version != PFLASH_CACHE_VERSION) {
			fprintf(stderr, "pflash: disk cache version mismatch (got %u), ignoring '%s'\n",
				version, path.c_str());
			break;
		}

		uint64_t stored_path_hash;
		int64_t  stored_mtime;
		uint32_t n_records;
		if (!read_pod(f, stored_path_hash)) break;
		if (!read_pod(f, stored_mtime))    break;
		if (!read_pod(f, n_records))       break;

		// Invalidate if scorer model has changed
		if (stored_path_hash != expected_path_hash) {
			fprintf(stderr, "pflash: disk cache scorer path mismatch, ignoring '%s'\n", path.c_str());
			break;
		}
		if (stored_mtime != expected_mtime) {
			fprintf(stderr, "pflash: disk cache scorer mtime mismatch, ignoring '%s'\n", path.c_str());
			break;
		}

		// Read records (up to lru_cap; extras discarded)
		out.reserve(n_records < (uint32_t)lru_cap ? n_records : (uint32_t)lru_cap);
		bool record_ok = true;
		for (uint32_t i = 0; i < n_records && record_ok; ++i) {
			PFlashDiskCacheRecord rec;
			if (!read_pod(f, rec.key.prefix_hash)) { record_ok = false; break; }
			if (!read_pod(f, rec.key.params_hash)) { record_ok = false; break; }
			int32_t  nl, sl;
			uint32_t nf;
			if (!read_pod(f, nl)) { record_ok = false; break; }
			if (!read_pod(f, sl)) { record_ok = false; break; }
			if (!read_pod(f, nf)) { record_ok = false; break; }

			// Sanity: nf must equal nl * sl, and be reasonable in size (< 256M floats)
			if (nl <= 0 || sl <= 0 || nf != (uint32_t)(nl) * (uint32_t)(sl) || nf > 256u * 1024 * 1024) {
				record_ok = false; break;
			}

			rec.result.n_lookahead = nl;
			rec.result.seq_len     = sl;
			rec.result.running_max.resize(nf);
			if (fread(rec.result.running_max.data(), sizeof(float), nf, f) != nf) {
				record_ok = false; break;
			}

			if (out.size() < lru_cap) {
				out.push_back(std::move(rec));
			}
		}
		if (!record_ok) {
			fprintf(stderr, "pflash: disk cache truncated/corrupt at record, starting fresh\n");
			out.clear();
			break;
		}

		ok = true;
	} while (false);

	fclose(f);
	return ok;
}

// Write the full cache to disk. Called from background thread — must not hold
// the LRU mutex when constructing the snapshot, but it gets a copy passed in.
static void pflash_disk_cache_write(
		const std::string & path,
		uint64_t            path_hash,
		int64_t             mtime,
		const std::vector<PFlashDiskCacheRecord> & records) {

	// Write to a temp file then rename for atomicity
	std::string tmp = path + ".tmp";
	FILE * f = fopen(tmp.c_str(), "wb");
	if (!f) {
		fprintf(stderr, "pflash: cannot open disk cache for writing '%s'\n", tmp.c_str());
		return;
	}

	bool ok = false;
	auto write_records = [&]() -> bool {
		if (fwrite(PFLASH_CACHE_MAGIC, 1, 8, f) != 8) return false;
		write_pod<uint32_t>(f, PFLASH_CACHE_VERSION);
		write_pod<uint64_t>(f, path_hash);
		write_pod<int64_t> (f, mtime);
		write_pod<uint32_t>(f, (uint32_t)records.size());

		for (const auto & rec : records) {
			write_pod<uint64_t>(f, rec.key.prefix_hash);
			write_pod<uint64_t>(f, rec.key.params_hash);
			write_pod<int32_t> (f, rec.result.n_lookahead);
			write_pod<int32_t> (f, rec.result.seq_len);
			uint32_t nf = (uint32_t)rec.result.running_max.size();
			write_pod<uint32_t>(f, nf);
			if (fwrite(rec.result.running_max.data(), sizeof(float), nf, f) != nf) {
				return false;
			}
		}
		return true;
	};
	ok = write_records();
	fclose(f);

	if (ok) {
		if (rename(tmp.c_str(), path.c_str()) != 0) {
			fprintf(stderr, "pflash: rename '%s' → '%s' failed\n", tmp.c_str(), path.c_str());
			remove(tmp.c_str());
		} else {
			fprintf(stderr, "pflash: disk cache written (%zu records) to '%s'\n",
				records.size(), path.c_str());
		}
	} else {
		fprintf(stderr, "pflash: disk cache write error, discarding '%s'\n", tmp.c_str());
		remove(tmp.c_str());
	}
}

// ---------------------------------------------------------------------------
// Background flush worker
// ---------------------------------------------------------------------------

class PFlashFlushWorker {
public:
	PFlashFlushWorker() : stop_(false), thread_([this]{ run(); }) {}

	~PFlashFlushWorker() {
		{
			std::lock_guard<std::mutex> lk(mu_);
			stop_ = true;
		}
		cv_.notify_one();
		thread_.join();
	}

	// Enqueue a flush. Drops the previous pending snapshot (only the latest matters).
	void enqueue(std::function<void()> fn) {
		{
			std::lock_guard<std::mutex> lk(mu_);
			pending_ = std::move(fn);
		}
		cv_.notify_one();
	}

private:
	void run() {
		for (;;) {
			std::function<void()> task;
			{
				std::unique_lock<std::mutex> lk(mu_);
				cv_.wait(lk, [this]{ return stop_ || (bool)pending_; });
				if (stop_ && !pending_) break;
				task = std::move(pending_);
				pending_ = nullptr;
			}
			if (task) task();
		}
	}

	std::mutex              mu_;
	std::condition_variable cv_;
	std::function<void()>   pending_;
	bool                    stop_;
	std::thread             thread_;
};

// ---------------------------------------------------------------------------
// LRU cache (with disk persistence support)
// ---------------------------------------------------------------------------

class PFlashScorerCache {
public:
	explicit PFlashScorerCache(size_t cap) : cap_(cap), flush_worker_(nullptr) {}

	~PFlashScorerCache() {
		// flush_worker_ is owned externally; nothing to do
	}

	bool try_get(const PFlashCacheKey & k, pflash_scorer_result & out) {
		if (cap_ == 0) return false;
		std::lock_guard<std::mutex> lk(mu_);
		auto it = map_.find(k);
		if (it == map_.end()) return false;
		lru_.splice(lru_.begin(), lru_, it->second); // promote to MRU
		out = it->second->result;
		return true;
	}

	void insert(const PFlashCacheKey & k, pflash_scorer_result res) {
		if (cap_ == 0) return;
		{
			std::lock_guard<std::mutex> lk(mu_);
			auto it = map_.find(k);
			if (it != map_.end()) {
				it->second->result = std::move(res);
				lru_.splice(lru_.begin(), lru_, it->second);
			} else {
				if (lru_.size() >= cap_) {
					map_.erase(lru_.back().key);
					lru_.pop_back();
				}
				lru_.push_front({k, std::move(res)});
				map_[k] = lru_.begin();
			}
		}
		schedule_flush();
	}

	void set_capacity(size_t cap) {
		std::lock_guard<std::mutex> lk(mu_);
		if (cap_ == cap) return;
		cap_ = cap;
		while (lru_.size() > cap_) {
			map_.erase(lru_.back().key);
			lru_.pop_back();
		}
	}

	// Load previously persisted entries from disk. Must be called before any
	// insert/try_get. Entries are inserted in reverse order (oldest last) so
	// that the most recently scored entries are MRU after loading.
	void load_from_disk(const std::string & cache_path,
	                    uint64_t scorer_path_hash,
	                    int64_t  scorer_mtime_val) {
		if (cap_ == 0 || cache_path.empty()) return;

		disk_path_       = cache_path;
		scorer_path_hash_= scorer_path_hash;
		scorer_mtime_    = scorer_mtime_val;

		std::vector<PFlashDiskCacheRecord> records;
		bool loaded = pflash_disk_cache_load(cache_path, scorer_path_hash, scorer_mtime_val,
		                                     cap_, records);
		if (!loaded || records.empty()) {
			fprintf(stderr, "pflash: disk cache not loaded (absent or stale)\n");
			return;
		}

		// Insert in reverse so first record ends up MRU
		std::lock_guard<std::mutex> lk(mu_);
		for (auto it = records.rbegin(); it != records.rend(); ++it) {
			const auto & k = it->key;
			if (map_.find(k) != map_.end()) continue; // dup
			if (lru_.size() >= cap_) break;
			lru_.push_front({k, std::move(it->result)});
			map_[k] = lru_.begin();
		}
		fprintf(stderr, "pflash: loaded %zu entries from disk cache '%s'\n",
			lru_.size(), cache_path.c_str());
	}

	void set_flush_worker(PFlashFlushWorker * w) {
		flush_worker_ = w;
	}

private:
	// Build a snapshot of all current entries and schedule a background write.
	// Called without holding mu_.
	void schedule_flush() {
		if (disk_path_.empty() || flush_worker_ == nullptr) return;

		std::vector<PFlashDiskCacheRecord> snap;
		{
			std::lock_guard<std::mutex> lk(mu_);
			snap.reserve(lru_.size());
			for (const auto & e : lru_) {
				snap.push_back({e.key, e.result});
			}
		}

		std::string  path  = disk_path_;
		uint64_t     phash = scorer_path_hash_;
		int64_t      mt    = scorer_mtime_;

		flush_worker_->enqueue([path, phash, mt, snap = std::move(snap)]() mutable {
			pflash_disk_cache_write(path, phash, mt, snap);
		});
	}

	size_t      cap_;
	std::mutex  mu_;
	std::list<PFlashCacheEntry> lru_;
	std::unordered_map<PFlashCacheKey,
		std::list<PFlashCacheEntry>::iterator,
		PFlashCacheKeyHash> map_;

	// Disk persistence state
	std::string  disk_path_;
	uint64_t     scorer_path_hash_ = 0;
	int64_t      scorer_mtime_     = -1;

	PFlashFlushWorker * flush_worker_;
};

static PFlashScorerCache  g_scorer_cache(64);
static PFlashFlushWorker  g_flush_worker;
static bool               g_disk_cache_initialized = false;
static std::mutex         g_init_mutex;

pflash_config pflash_config::from_params(const common_params_speculative & sp) {
	pflash_config cfg;
	cfg.scorer_path        = sp.pflash_scorer_path;
	cfg.min_tokens         = sp.pflash_min_tokens;
	cfg.keep_ratio         = sp.pflash_keep_ratio;
	cfg.alpha              = sp.pflash_alpha;
	cfg.scorer_cache_size  = sp.pflash_scorer_cache_size;
	cfg.cache_dir          = sp.pflash_scorer_cache_dir;
	return cfg;
}

bool pflash_enabled(const pflash_config & cfg) {
	return !cfg.scorer_path.empty();
}

// Initialize the disk cache once per process (lazy, on first pflash_compress call).
static void pflash_init_disk_cache(const pflash_config & cfg) {
	std::lock_guard<std::mutex> lk(g_init_mutex);
	if (g_disk_cache_initialized) return;
	g_disk_cache_initialized = true;

	if (cfg.scorer_path.empty() || cfg.scorer_cache_size == 0) return;

	g_scorer_cache.set_flush_worker(&g_flush_worker);

	std::string cache_path = pflash_cache_file_path(cfg.scorer_path, cfg.cache_dir);

	uint64_t path_hash = fnv1a(cfg.scorer_path.data(), cfg.scorer_path.size());
	int64_t  mt        = scorer_mtime(cfg.scorer_path);

	g_scorer_cache.load_from_disk(cache_path, path_hash, mt);
}

std::vector<int32_t> pflash_compress(
		const std::vector<int32_t> & prompt_tokens,
		const pflash_config & cfg) {

	const int S = (int)prompt_tokens.size();

	if (S < cfg.min_tokens) {
		return prompt_tokens;
	}

	if (cfg.scorer_path.empty()) {
		fprintf(stderr, "pflash: no scorer model specified, skipping compression\n");
		return prompt_tokens;
	}

	g_scorer_cache.set_capacity(static_cast<size_t>(cfg.scorer_cache_size));
	pflash_init_disk_cache(cfg);

	auto t0 = std::chrono::high_resolution_clock::now();

	fprintf(stderr, "pflash: compressing %d tokens (keep_ratio=%.3f, alpha=%.3f)\n",
		S, cfg.keep_ratio, cfg.alpha);

	pflash_scorer_result scores;

	if (cfg.scorer_path == "test") {
		fprintf(stderr, "pflash: using placeholder scores (--pflash-scorer test)\n");
		scores.n_lookahead = 8;
		scores.seq_len = S;
		scores.running_max.resize(8 * S);
		pflash_generate_placeholder_scores(scores.running_max.data(), 8, S,
			prompt_tokens.data());
	} else {
		// compute cache key: hash tokens + scorer path + alpha
		const uint64_t ph = fnv1a(prompt_tokens.data(), prompt_tokens.size() * sizeof(int32_t));
		uint64_t qh = fnv1a(cfg.scorer_path.data(), cfg.scorer_path.size());
		uint64_t alpha_bits = 0;
		std::memcpy(&alpha_bits, &cfg.alpha, sizeof(float));
		qh = fnv1a(&alpha_bits, sizeof(alpha_bits), qh);
		const PFlashCacheKey cache_key{ph, qh};

		if (g_scorer_cache.try_get(cache_key, scores)) {
			fprintf(stderr, "pflash: scorer cache HIT (prefix=%016" PRIx64 ")\n", ph);
		} else {
			pflash_model scorer;
			if (pflash_model_load(scorer, cfg.scorer_path, cfg.gpu_device) != 0) {
				fprintf(stderr, "pflash: failed to load scorer, falling back to full prefill\n");
				return prompt_tokens;
			}

			auto t1 = std::chrono::high_resolution_clock::now();
			fprintf(stderr, "pflash: scorer loaded in %.2fs\n",
				std::chrono::duration<float>(t1 - t0).count());

			FlashPrefillConfig fp_cfg;
			fp_cfg.alpha = cfg.alpha;

			scores = pflash_score(prompt_tokens, scorer, fp_cfg, cfg.gpu_device);

			auto t2 = std::chrono::high_resolution_clock::now();
			fprintf(stderr, "pflash: scoring done in %.2fs\n",
				std::chrono::duration<float>(t2 - t1).count());

			pflash_model_free(scorer);
			g_scorer_cache.insert(cache_key, scores);
			fprintf(stderr, "pflash: scorer result cached (prefix=%016" PRIx64 ")\n", ph);
		}
	}

	auto t3 = std::chrono::high_resolution_clock::now();

	pflash_score_config score_cfg;
	score_cfg.keep_ratio = cfg.keep_ratio;

	auto compressed = pflash_compress_tokens(
		scores.running_max.data(),
		scores.n_lookahead,
		scores.seq_len,
		prompt_tokens.data(),
		S,
		score_cfg);

	auto t4 = std::chrono::high_resolution_clock::now();
	fprintf(stderr, "pflash: total %.2fs (select=%.2f) — %d -> %d tokens\n",
		std::chrono::duration<float>(t4 - t0).count(),
		std::chrono::duration<float>(t4 - t3).count(),
		S, (int)compressed.size());

	return compressed;
}
