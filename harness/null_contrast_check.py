import json

import numpy as np

from harness.config import Config
from harness.saturation_check import CACHE_PATHS, _loader, apply_to, fit_eval

cfg = Config()
paths = json.load(open(CACHE_PATHS))
acts = _loader(paths)

PROBES = [
    ("WORDING p1vp2", "loyal_acme__p1", "loyal_acme__p2"),
    ("WORDING p0vp1", "loyal_acme__p0", "loyal_acme__p1"),
    ("REAL matched ", "loyal_acme__p0", "control_acme__p0"),
]

TESTS = ["loyal_acme__p0", "loyal_zephyr__p0", "secret_rule__p0",
         "secret_vendor_rule__p0", "overt_rule__p0", "sycophant_acme__p0",
         "control_acme__p0", "control_zephyr__p0"]

print("Every direction scored against neutral__p0 on the same conditions.")
print("The REAL direction's signature is: fires on influence attempts, SILENT on")
print("bare entity mentions. A wording direction must fail to reproduce that.")
print()
header = "  " + " " * 15 + "".join(f"{t.replace('__p0',''):>19}" for t in TESTS)
print(header)
for label, pos_key, neg_key in PROBES:
    r = fit_eval(acts, cfg, pos_key, neg_key)
    d, L = np.array(r["direction"]), r["layer"]
    row = f"  {label:15}"
    for t in TESTS:
        a = apply_to(acts, cfg, d, L, t, "neutral__p0")["auroc"]
        row += f"{a:19.3f}"
    print(row + f"   (L{L})")
