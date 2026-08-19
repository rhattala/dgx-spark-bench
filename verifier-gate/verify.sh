#!/usr/bin/env bash
# One command. Runs a verified best-of-3 and refuses to print a number it cannot trust.
#
#   ./verify.sh /path/to/llm-as-a-verifier
#
# Defaults to K=1 with all three criteria. K=1 performed the same as K=2 under
# every reconstruction we tried; the earlier '40 shuffling' ablation that also
# claimed a single criterion was worse has been WITHDRAWN (it fabricated 20.8%
# of its inputs). See verifier-gate/README.md.
set -uo pipefail
REPO="${1:-.}"
WORKERS="${WORKERS:-4}"
cd "$REPO" || { echo "usage: verify.sh /path/to/llm-as-a-verifier"; exit 2; }

SRC="$(cd "$(dirname "$0")" && pwd)"
for f in verify_run.py run_and_gate.sh; do
  if [ ! -f "$f" ]; then
    cp "$SRC/$f" . || { echo "could not copy $f from $SRC — aborting"; exit 2; }
  fi
done

echo "==> running best-of-3, K=1, all criteria, ${WORKERS} workers"
LLMV_TIMEOUT="${LLMV_TIMEOUT:-1800}" WORKERS="$WORKERS" \
  ./run_and_gate.sh 2>&1 | tail -40
rc=${PIPESTATUS[0]}

echo
if [ "$rc" -eq 0 ]; then
  echo "==> GATE PASSED — the score above came from measured data."
else
  echo "==> GATE DID NOT PASS (rc=$rc). Do NOT publish the number above."
  echo "    rc=1 something is provably wrong (missing scores, or ties that were never measured)"
  echo "    rc=2 nothing provably wrong, but completeness could not be established"
fi
exit $rc
