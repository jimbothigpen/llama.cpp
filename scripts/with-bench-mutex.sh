#!/usr/bin/env bash
# with-bench-mutex.sh — per-host advisory file lock for llama-bench
# and compile-tier jobs.
#
# Wrap any of the following with this script:
#   - llama-bench (any backend)
#   - cmake / ninja / make / "cmake --build" / "sudo ninja install"
#
# Compile-tolerant jobs (llama-perplexity, llama-quantize, llama-imatrix)
# do NOT need this wrapper.
#
# Convention: yggdrasil-context/conventions/bench-compile-mutex.md
#
# Usage:
#   with-bench-mutex.sh <command> [args...]
#
# Env overrides:
#   YGGDRASIL_BENCH_LOCK — override lock file path (default
#     /var/lock/yggdrasil-bench.lock; falls back to /tmp if unwritable).

set -euo pipefail

LOCK_FILE="${YGGDRASIL_BENCH_LOCK:-/var/lock/yggdrasil-bench.lock}"

# Probe writability; fall back to /tmp if /var/lock is read-only.
if ! { : > "$LOCK_FILE"; } 2>/dev/null; then
  LOCK_FILE=/tmp/yggdrasil-bench.lock
fi
WHO_FILE="${LOCK_FILE%.lock}.who"

exec 9>"$LOCK_FILE"
flock 9

# Write the holder sidecar. The EXIT trap removes it on normal/abnormal
# exit (SIGTERM, kill, exception). SIGKILL leaves it stale, but the
# flock itself releases correctly because the kernel tracks fd 9.
#
# IMPORTANT: don't use `exec "$@"` here — exec replaces the shell
# process before bash gets a chance to run the EXIT trap, so .who
# would never be cleaned up. Running as a child + propagating $? is
# the correct shape.
printf 'host=%s\npid=%s\ncmd=%s\nsince=%s\n' \
  "$(hostname)" "$$" "$*" "$(date -Iseconds)" > "$WHO_FILE"
trap 'rm -f "$WHO_FILE"' EXIT

"$@"
exit $?
