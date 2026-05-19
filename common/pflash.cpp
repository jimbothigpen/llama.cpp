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

class PFlashScorerCache {
public:
	explicit PFlashScorerCache(size_t cap) : cap_(cap) {}

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
		std::lock_guard<std::mutex> lk(mu_);
		auto it = map_.find(k);
		if (it != map_.end()) {
			it->second->result = std::move(res);
			lru_.splice(lru_.begin(), lru_, it->second);
			return;
		}
		if (lru_.size() >= cap_) {
			map_.erase(lru_.back().key);
			lru_.pop_back();
		}
		lru_.push_front({k, std::move(res)});
		map_[k] = lru_.begin();
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

private:
	size_t      cap_;
	std::mutex  mu_;
	std::list<PFlashCacheEntry> lru_;
	std::unordered_map<PFlashCacheKey,
		std::list<PFlashCacheEntry>::iterator,
		PFlashCacheKeyHash> map_;
};

static PFlashScorerCache g_scorer_cache(64);

pflash_config pflash_config::from_params(const common_params_speculative & sp) {
	pflash_config cfg;
	cfg.scorer_path        = sp.pflash_scorer_path;
	cfg.min_tokens         = sp.pflash_min_tokens;
	cfg.keep_ratio         = sp.pflash_keep_ratio;
	cfg.alpha              = sp.pflash_alpha;
	cfg.scorer_cache_size  = sp.pflash_scorer_cache_size;
	return cfg;
}

bool pflash_enabled(const pflash_config & cfg) {
	return !cfg.scorer_path.empty();
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
