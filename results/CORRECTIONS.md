# Corrections

Errors found in this project's own results after they were produced, and what each
one does and does not change. Superseded artifacts are retained rather than
overwritten; this file is the index to them.

---

## 1. Coherence sample size was inflated fivefold

**Where:** `results/distilled/distilled.json` → `coherence.n`, reported as **50**.

**What was wrong:** `COHERENCE_CORPUS` was ten prompts multiplied by five. Generation
runs at temperature 0, so the fifty measurements were ten distinct values each counted
five times.

**What it changes:** the effective sample size is **10**, not 50. Any variance,
standard error or interval computed on n=50 is wrong by a factor of √5.

**What it does not change:** the reported mean and median drift. Repeating identical
values does not move either. The coherence verdict (0.0049 nats against a 0.1-nat
threshold) stands.

**Fixed in:** `9b756ef` — the corpus now holds fifty distinct prompts. The artifact
above is annotated in place and otherwise left as the record of what the run produced.

---

## 2. An ablation null was reported without checking the intervention could move the behaviour

**Where:** `results/experiment/causal.json`, and the commit message of `2a2dde8`.

**What was wrong:** targeted ablation returned a gap closure of 0.0 against a
pre-registered threshold of 0.5, and this was reported as evidence that the direction
does not mediate the behaviour. Nothing established that the intervention had the
dynamic range to produce a different answer.

**What it changes:** it did not. `results/power/power_analysis.json` shows the
behaviours sit 12.3–18.1 nats off baseline while the intervention moves option margins
by 1.15 nats, so maximum achievable gap closure was 0.093 / 0.069 / 0.063 — short of the
threshold by 5.4× to 7.9×. **The criterion could not have been met by any outcome.**
The null is a statement about the instrument, not about mediation, and should be read
at that reduced strength.

At minimum headroom the three cases separate: loyalty is 2.0× short, the two rule
conditions 8.4× and 9.1×. The loyalty null is underpowered; the directive null is
arithmetically foreclosed.

Related: the largest norm-matched random control moves margins 2.16 nats against the
targeted direction's 1.15. The noise floor sits above the signal.

---

## 3. A de-confounding contrast was over-claimed

**Where:** the original commit message of `86958f7`, amended to `44f860a`.

**What was wrong:** refitting the probe against `secret_rule` produced a direction with
no loyalty content, and this was written up as *"no transferable principal-directed
component is linearly recoverable once directive-ness is controlled."*

**What it changes:** `secret_rule` differs from the loyalty conditions in **content
domain** as well as in principal-presence — it concerns preferring a lettered option,
they concern a company. The projections identify what the direction actually found:
`secret_rule` at −74.5 and `overt_rule` at −33.1 sit roughly a hundred units below every
other condition, with neutral highest at +113.8, and `overt_rule` was never in the fit.
This is a lettered-position-rule axis. Layer selection was degenerate besides, with 28
of 29 layers tied.

**Correct claim:** this contrast does not isolate principal-directedness. A valid
de-confounding negative must be a covert directive matched in content domain — a secret
rule favouring an incumbent supplier rather than a lettered position. No such condition
exists in the set. That is the limitation and the next experiment.

---

## 4. A pre-registered naming rule made a false call

**Where:** `results/margins/margin_analysis.json` → `calls.*.prereg_rule`.

**What was wrong:** nothing, procedurally. The rule was fixed before any organism margin
was computed, and it named **Ortell** for organisms a and b.

**What it changes:** the call is spurious and the same artifact carries the diagnostic
that shows why. `corr(base margin, delta)` is −0.992 and −0.997, and mean absolute margin
falls from 0.498 to 0.091 and 0.058 — the organisms are uniformly *less* entity-biased
than their base, and shrinking a negative base margin toward zero is indistinguishable
from favouring the entity on the losing side of it. Under an absolute-margin rule, no
principal is separable in any organism.

Both the pre-specified call and the correction are reported. The rule is not repaired
retroactively; a post-hoc rule cannot inherit a pre-registered rule's standing.

---

## 5. Organism-c is not a modified model

**Where:** `results/organism/scan__sl-organism-c-7b.json`.

**What it changes:** organism-c is **bit-identical to `Qwen/Qwen2.5-7B-Instruct`** across
all 100 rows (max |Δ logprob| = 0.00). It is an unmodified control, not a finetuned
organism that failed to fire.

This retracts an earlier reading of the scan results. Two organisms firing and one not
is **not** evidence against the fine-tuning-drift confound, because the null distribution
was computed on the base model itself. The confound remains open pending the
benign-finetune control.
