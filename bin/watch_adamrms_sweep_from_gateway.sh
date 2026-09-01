#!/bin/bash
set -euo pipefail

JZ_HOST="${JZ_HOST:-ulf36rc@jean-zay.idris.fr}"
JZ_KEY="${JZ_KEY:-${HOME}/.ssh/id_ed25519}"
REMOTE_REPO="${REMOTE_REPO:-/lustre/fswork/projects/rech/qps/ulf36rc/spectron_mechanistic}"
POLL_SECONDS="${POLL_SECONDS:-900}"
MAX_TICKS="${MAX_TICKS:-384}"
LOCK_DIR="${LOCK_DIR:-/tmp/spectron_adamrms_sweep_watch.lock}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another gateway watchdog owns ${LOCK_DIR}" >&2
  exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

for ((tick = 1; tick <= MAX_TICKS; tick++)); do
  echo "gateway_tick=${tick} time=$(date -Is)"
  output=$(ssh \
    -i "${JZ_KEY}" \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -o IdentityAgent=none \
    "${JZ_HOST}" \
    "python3 '${REMOTE_REPO}/bin/watch_adamrms_sweep.py' --repo '${REMOTE_REPO}'" \
  ) || true
  printf '%s\n' "${output}"
  if grep -q '"phase": "awaiting_std1_approval"' <<< "${output}"; then
    echo "gateway_watchdog_done time=$(date -Is)"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done

echo "gateway_watchdog_expired time=$(date -Is)" >&2
exit 1
