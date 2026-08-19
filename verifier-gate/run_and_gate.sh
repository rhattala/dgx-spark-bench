#!/usr/bin/env bash
# Benchmark -> gate -> remediate -> acceptance, per the reviewed plan.
#
# The remediation loop matters: max_retries=0 means any transient failure loses
# that call, and a lost call is silently scored 0.5/0.5 for the run. The cache
# never stores those, so re-running re-scores ONLY the missing keys. A transient
# 503 is therefore one more cheap pass, NOT a broken run.
set -uo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-4}"
MAXPASS="${MAXPASS:-4}"
CACHE="cache/cache_terminal_bench_2.1_mswea_deepseek_bo3.json"

echo "=== T-0 PRECONDITIONS ==="
if [ -s "$CACHE" ]; then
  if [ "${RESUME:-0}" != "1" ]; then
    echo "  cache is NON-EMPTY at T-0: $CACHE"
    echo "  protocol requires an empty cache for a FRESH run."
    echo "  If this is a deliberate resume, re-launch with RESUME=1. ABORTING."
    exit 2
  fi
  # Deliberate resume. The guard is not weakened: an intentional resume must
  # still prove the surviving cache is CLEAN, because poisoned entries are
  # cached and the resume path only re-scores MISSING keys -- it would never
  # repair them.
  POISON=$(python3 -c "import json;c=json.load(open('$CACHE'));print(sum(1 for v in c.values() if v.get('score_A')==0.5 or v.get('score_B')==0.5))")
  if [ "$POISON" != "0" ]; then
    echo "  RESUME REFUSED: $POISON cached entr(ies) hold an exact 0.5."
    echo "  Resume only re-scores MISSING keys, so these would survive to the"
    echo "  final result. Purge them first. ABORTING."
    exit 2
  fi
  N=$(python3 -c "import json;print(len(json.load(open('$CACHE'))))")
  echo "  RESUME: $N clean entries retained, $((576-N)) to score, 0 poisoned"
else
  echo "  cache empty: ok"
fi
R=$(curl -s -m 10 http://spark-1:8888/metrics | grep '^vllm:num_requests_running{' | awk '{print $2}')
[ "$R" = "0.0" ] || { echo "  engine not idle (running=$R). ABORTING."; exit 2; }
echo "  engine idle: ok"

for pass in $(seq 1 "$MAXPASS"); do
  echo
  echo "=== PASS $pass — run_bo3 --max-workers $WORKERS ==="
  .venv/bin/python scripts/run_bo3.py --max-workers "$WORKERS" 2>&1 | tr '\r' '\n' | grep -vE "it/s\]|Scoring:.*\|"
  echo
  echo "=== PASS $pass — gate ==="
  .venv/bin/python verify_run.py --strict; rc=$?
  echo "  gate rc=$rc"
  [ "$rc" -eq 0 ] && break
  [ "$pass" -eq "$MAXPASS" ] && { echo "  gate never reached PASS in $MAXPASS passes — STOPPING, do not publish."; exit 1; }
  echo "  re-running: resume re-scores ONLY the missing keys"
done

echo
echo "=== ACCEPTANCE: replay the completed cache through run.py against a DEAD port ==="
echo "    (a complete cache makes zero API calls; an incomplete one fails loudly)"
OPENAI_BASE_URL=http://127.0.0.1:9/v1 OPENAI_API_KEY=EMPTY \
  .venv/bin/python scripts/run_bo3.py 2>&1 | tr '\r' '\n' | grep -vE "it/s\]|Scoring:.*\|" | tail -30
