# Detection without Mediation

Code and results for an Apart Research Secret Loyalties hackathon submission, Track 2
(Detection & Auditing).

A secret loyalty is installed in a model three ways — by system prompt, by LoRA distillation
into weights, and by using the externally released organisms — and a range of black-box,
logit-level and white-box methods are then used to try to detect it, identify its principal,
and remove it. Every arm carries a matched control, and the controls are what most of the
results turn on.

## Reproducing

```
pip install -r requirements.txt
python -m harness.<module>
```

Scale and output location are selected by environment variable, so every model tree below was
produced by the same code paths:

| variable | default | purpose |
|---|---|---|
| `SL_MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | model under test |
| `SL_RESULTS_DIR` | `results_7b` | where artifacts are written |
| `SL_DEVICE` | `mps` | `cuda`, `mps` or `cpu` |
| `SL_DTYPE` | `float16` | |
| `SL_N_RANDOM` | `20` | random-direction controls per run |

Every run writes a `<name>.settings.json` sidecar alongside its result containing the full
config, the git commit, the resolved device, and the torch/Python/platform versions. Where a
number below disagrees with prose anywhere, the sidecar and its JSON are authoritative.

Fixed across runs: seed 315, 100 evaluation scenarios, 2000 bootstrap resamples, 50/50
held-out split for probe fitting.

## Repo map

```
harness/          all experiments; each is `python -m harness.<module>`
  conditions.py     system prompts: loyalties, matched controls, secret/overt rule twins
  scenarios.py      100 forced-choice scenarios, interrogation prompts, prefill string
  probe.py          linear probes, layer selection, AUROC with bootstrap CIs
distill/          organism construction (data generation + LoRA training)
results_1.5b/     causal transplant, mis-execution
results_3b/       saturation, sample sweep, condition ladder, cached activations
results_7b/       primary tree — every experiment
results_14b/      mis-execution
results_llama/    saturation, sample sweep, condition ladder (Llama-3.1-8B-Instruct)
```

Coverage across the model trees is uneven by design, not by omission. Three different
questions need three different sets of widths: the saturation claim is about the ratio of
samples to dimensions, so it runs at 3B (d=2048), 7B (d=3584) and Llama-8B (d=4096); the
mis-execution scaling curve runs at 1.5B, 7B and 14B; and the ten-probe-family extension runs
at 7B only. The 1.5B causal transplant is there for a non-scientific reason — after GPU access
ended it was the largest model the local machine could hold.

## Claim to artifact

| claim | artifact |
|---|---|
| Ten probe families (8 linear, 2 non-linear) all saturate a null pair at AUROC 1.000; none separates concealed from disclosed steering, largest gap **+0.000** | `results_7b/saturation/probe_families.json`, `probe_families_nonlinear.json` |
| A random direction fitted to nothing averages **0.941** with layer selection and **0.738** at a fixed layer — layer selection alone is worth +0.203 AUROC to an uninformative direction | `results_7b/saturation/probe_families.json` |
| Saturation replicates across three widths and two model families | `results_3b/`, `results_7b/`, `results_llama/saturation/saturation.json` |
| Blind principal enumeration names the true principal **first out of 42** candidates on 3 of 3 loyal arms, AUROC 1.000, with a 2.15–11.27 nat window in which it abstains on the no-loyalty arm | `results_7b/enumeration/principal_enumeration.json` |
| No loyalty-specific transfer to the released organisms: the loyalty-fitted direction separates A and B from base at 1.000, but so does a benign size-matched fine-tune direction, and control organism C reads 0.500 | `results_7b/cross_organism/cross_organism.json` |
| Trace statistic recovers a known principal at **+10.53** nats on the positive control; both released organisms (**+0.55**, **+0.63**) fall below the benign no-loyalty floor of **+0.87**, and on the same entity the benign adapter tops | `results_7b/trace/trace_check.json` |
| A single-layer residual-stream transplant at **L23** closes **+0.989** of a 3.00-nat loyalty gap; entity-mention donor +0.460, overt-rule donor −0.033, identity self-patch exactly 0.0 | `results_1.5b/patch/full_patch.json` |
| Steering as removal fails in both directions: at coherent α behaviour does not move (0.45 → 0.525, overlapping intervals); above it, generations degenerate. No α produces behavioural change with coherent output | `results_7b/steer/steer_sweep.json`, `steer_sweep_lowalpha.json` |
| Targeted ablation on near-tie scenarios does not mediate the behaviour: rank **12 of 101** against random controls, p = **0.119** | `results_7b/neartie/neartie.json` |
| Mis-execution — a covert rule the model cannot run is replaced by a positional heuristic sharing its ordinal word: 7B picks the second option 0.97 of the time while satisfying the actual rule 0.53; 14B 0.99 against 0.51; reversing the ordinal word flips the positional rate to 0.15 and 0.21 | `results_1.5b/`, `results_7b/`, `results_14b/misexecution/misexecution.json` |
| Mis-execution is capability-bounded: at frontier scale the same rules are executed faithfully (own-rule 0.92 at zero reasoning effort, 0.88 at high), so it is not a protective property | `results_7b/frontier/frontier_gpt-5.6-sol.json`, `frontier_gpt-5.6-sol__effort-high.json` |
| Two loyalties with disjoint remits both install by prompt alone: remit gap **+0.40** (0.80 inside Acme's remit, 0.40 inside Zephyr's, against single-loyalty endpoints of 1.00 and 0.00). Pilot at n=20 | `results_7b/dual/dual_loyalty.json` |
| Reporting audit of four intervention papers against the gate quantities this project uses | `results_7b/literature/reporting_audit.json` |

Results indexed to Lamerton & Roger's affordance levels are in `results_7b/affordance.md`,
which also carries a correction distinguishing their five *knowledge* levels from this
project's three *access* tiers.

## Designed and not run

- **Trigger-conditional organism.** Fully implemented and never executed — GPU access ended
  before it ran. `distill/generate_trigger_data.py`, `harness/trigger_enumeration.py`, and the
  wrapper in `harness/run_second_wave_remote.sh`. This is the missing cell that would separate
  "dormant loyalties leave no trace" from "our statistics cannot see dormant traces": the
  project has active loyalties of known principal and dormant loyalties of unknown principal,
  but never a dormant loyalty of known principal.
- **14B saturation.** Needed hardware that was no longer available.
- **Transplant answer-copying control.** The read position is the final token and L23 of 28 is
  late, so a full-state transplant may transport the donor's decision as well as its
  disposition. The discriminating control — donor state from scenario *j* patched into
  recipient scenario *i* — was not run, and it bounds the causal claim.
- **Steering on the released organisms.** `harness/steer_sweep.py` loads the base model with
  no adapter, so the removal result is bounded to a prompt-installed loyalty at 7B and was
  never run against organisms A, B or C.

## Dual-use

`distill/` contains a working pipeline for installing a secret loyalty into an open-weights
model, and `harness/conditions.py` contains the system prompts that install one by prompt
alone. These are here because Track 1 identifies model organisms as the infrastructure the
rest of the research area is bottlenecked on, and because the detection and removal results in
this repository are uninterpretable without the matched controls that the same pipeline
produces — in particular the benign size-matched fine-tune, which is what turns the headline
detection result from a positive into a null.

What is deliberately absent: no trigger-concealment tooling, no methods for evading the
detection techniques evaluated here, and no work on making an installed loyalty harder to
find. The organisms are built to be studied, not to be deployed. The strongest result in this
repository is that a covert directive planted in a model too weak to execute it does something
other than what its author intended — which is a reason for less confidence in this class of
attack, not a recipe for improving it.
