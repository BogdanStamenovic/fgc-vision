"""usage: gt_set.py KEY id=L [id=L ...]  -- record hand labels"""
import json, sys
from pathlib import Path
p = Path(f"/home/bodas/data/fgc-vision/labels/gt_{sys.argv[1]}.json"); d = json.loads(p.read_text())
for kv in sys.argv[2:]:
    k, v = kv.split("="); d["fragments"][k] = v
p.write_text(json.dumps(d, indent=1)); print(len(d["fragments"]), "labelled")
