#!/usr/bin/env bash
# Borrow the GPU from cvoiced (Bogdan's TTS server) during development.
# Only unloads when cvoiced reports phase "ready" (idle). Never kills it; it reloads on
# its next request. Server address and token come from ~/.config/cvoice/config.toml.
set -euo pipefail
read -r URL TOKEN < <(python3 - <<'PY'
import os, tomllib
c = tomllib.load(open(os.path.expanduser("~/.config/cvoice/config.toml"), "rb"))["server"]
print(f"http://{c['host']}:{c.get('port', 8760)}", c.get("token") or "-")
PY
)
case "${1:-status}" in
  status) curl -s "$URL/status"; echo; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader ;;
  borrow)
    phase=$(curl -s "$URL/status" | python3 -c "import sys,json;print(json.load(sys.stdin)['phase'])")
    case "$phase" in ready|idle) ;; *) echo "cvoiced busy ($phase), not unloading" >&2; exit 1 ;; esac
    curl -s -X POST -H "Authorization: Bearer $TOKEN" "$URL/unload"; echo ;;
  restore) curl -s -X POST -H "Authorization: Bearer $TOKEN" "$URL/load"; echo ;;
esac
