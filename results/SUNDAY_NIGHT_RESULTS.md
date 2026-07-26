# Sunday-night results — what changed after the skeleton was drafted, and where each lands

Every number verified programmatically against its JSON before entering this file. Items
marked ⏳ were in flight on the pod when it went down and are relaunchable via
`harness/run_second_wave_remote.sh`.

| result | number | artifact | lands in skeleton |
|---|---|---|---|
| ten probe families all saturate | worst null AUROC **1.000** across 8 linear + MLP + k-NN | `saturation/probe_families.json`, `probe_families_nonlinear.json` | §4.4 (failure 4) — replaces "diff-of-means only" caveat; §4.4a becomes "ten families" |
| random-direction floor | **0.738** fixed layer, **0.941** with layer selection, **0.994** on the headline contrast | `saturation/probe_families.json` | §4.4 — the sharpest single sentence of failure 4; layer selection alone is +0.20 |
| concealment disconfirming test | gap **+0.000** in all ten families | both probe_families artifacts | §4.2 + the inversion's scope paragraph — "weak probe" objection closed for linear+MLP+kNN |
| null-contrast lead resolved | cos **+0.478/+0.473** with real axis; dies residualised; survives 20/20 random removals | `saturation/null_contrast_lead.json` | §4.4 (correction 9 follow-up) — signature is geometry of one condition |
| near-tie settled at 100 controls | rank **12/101**, p = **0.119** (was 2/21, 0.095) | `neartie/neartie.json` (n20 retained) | §4.3a — "no causal result" now on a resolved number; SELF_AUDIT #17/#18 |
| principal enumeration | **3/3** organisms, rank 1 of **42**, z ≈ 20, threshold window **(2.15, 11.27)** nats | `enumeration/principal_enumeration.json` | §4.6/§4.7 + affordance black-box row — the positive result; scope: always-on only |
| mis-execution dissociates by failure type — **five scales** | ordinal-word flip flat at **0.80/0.82/0.78** (1.5B/7B/14B), gone at frontier (0.02); info-absent shift decays **0.63–0.90 → 0.21–0.50 → ~0**, dead between 7B and 14B; frontier+reasoning unchanged (0.88 own-rule, 9.6k reasoning tokens); own-rule ~0.5 at every live scale, so compliance alone never shows either threshold | `results_15b/misexecution/`, `results_14b/misexecution/`, `frontier/frontier_gpt-5.6-sol__effort-high.json` | §4.3 ladder + SELF_AUDIT #16 — two thresholds, each anchored at both ends; Track-3: CoT doesn't surface it |
| literature reporting audit | **0 of 4** intervention papers report all three gate quantities; **0 of 4** run random-direction controls; MacDiarmid shows both floor phenomena (best of 1000 random dirs = **95.3%**) | `literature/reporting_audit.json` | §2 Related Work + §5.1b — the field-level claim; propose the three-number reporting row |
| calibration table | worst null ≥0.95 mean from **n=10/side** (7B, 8B), n=20 (3B) | `sweep/calibration.md` | §4.4 or appendix C — the lookup any team can read their own numbers against |
| L&R Table 3 re-read | Level-4 detection **0–17%**, prefill rows 0% everywhere, n=30/cell | SELF_AUDIT #19 | §2 + enumeration contrast — cite their knowledge levels correctly |
| affordance attribution corrected | their §4.3 levels are knowledge, not access; tiers are ours | `affordance.md` header + SELF_AUDIT #19 | §2 Related Work MUST state the axis distinction |
| ✅ **full-state transplant — the causal result** | single layer **L23 closes +0.989** of the 3.0-nat gap (onset L19, nothing L3–L15, 20/40 flips); entity-mention donor **+0.460**, overt-rule donor **−0.033**, identity exactly 0.0, single-layer beats the all-layer ceiling (+0.793) | `results_15b/patch/full_patch.json` (1.5B; 7B run lost with pod) | §4.3a becomes a positive: the state carries loyalty, every ablation null was rank-one failure; §4.7 inversion sharpens to "the costliest affordance is where the standard method throws away what the access bought" |
| ✗ 14B saturation | **not run — pod terminated before the sweep started.** The regime claim keeps three widths (d = 2048/3584/4096) | module ships (`probe_families.py` under any `SL_RESULTS_DIR`) | §4.4 limitation sentence: "a fourth width (d=5120) was cached but lost to compute" |
| ✗ trigger organism + rate curve | **not run — needs training compute.** Full pipeline ships unrun; named as the designed-and-not-run experiment the enumeration scope calls for | `distill/generate_trigger_data.py`, `harness/trigger_enumeration.py`, `harness/run_second_wave_remote.sh` | §5.3/Future Work — one paragraph, with the gate criteria pre-registered in the module |

## Related Work sentences the new results require

- **Hewitt & Liang (2019) control tasks must be cited** as the prior art for "probe accuracy
  can be vacuous"; the delta is: paraphrase nulls + the random-direction floor + the
  layer-selection increment + deployment framing (concealed vs disclosed), measured at the
  n and d the field actually uses.
- **MacDiarmid et al.** is corroboration, not competition: their 95.3%-of-1000-random-dirs
  and generic-contrast findings are our two floor results at frontier scale, and they
  published them without proposing them as a validation bar — which is the gap the gates fill.
- **Lamerton & Roger**: knowledge levels ≠ access tiers (SELF_AUDIT #19). Their Level-4
  0–17% with prefill at 0% everywhere vs our 3/3-at-42-candidates is the
  active-vs-dormant contrast, not a method win.

## The one-sentence upgrades available to the abstract

- "diff-of-means directions" → "ten probe families including non-linear"
- "a contrast carrying no loyalty information reproduces the signature" → add "because it
  partially recovers the same axis — and a random direction with layer selection reads 0.94"
- add: "none of four prominent steering papers publishes the three numbers needed to run
  these checks; we propose them as a reporting standard"
- the trigger-rate curve did not run (compute gone); the abstract keeps the qualitative
  form — "enumeration is free when the loyalty is active; the released organisms show one
  dormant candidate was already too many" — and the pipeline ships for whoever runs it
