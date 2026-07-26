# Corrections

Errors found in this project's own results after they were produced, and what each
one does and does not change. Superseded artifacts are retained rather than
overwritten; this file is the index to them.

Four of these are the same failure: a standard-playbook criterion returned a confident
answer that no outcome could have contradicted. Corrections 2 (an unreachable ablation
threshold), 4 (a naming rule that fires under uniform debiasing), 6 (a probe that separates
perfectly on the wrong variable) and 8 (a validation metric saturated at ceiling) are
instances of it, not anecdotes. The checks that catch them are packaged as two runnable
modules — `harness/reachability.py`, which needs no model or activations and works on any
team's numbers, and `harness/saturation_check.py`, which needs only cached activations:

```
python -m harness.reachability --examples          # this project's own numbers
python -m harness.reachability --behaviour-gap 16.6 --intervention 1.15 \
       --threshold 0.5 --random 2.16 --target-delta 1.549 --off-target-delta 1.577
python -m harness.saturation_check                # is your headline AUROC informative?
```

It exits non-zero when a check fails, so it works as a pre-write-up gate. Worked output is
`results/power/reachability_examples.json`. Note that the two causal experiments in this
project disagree on one check — the earlier all-layer ablation fails the noise floor
(targeted 1.15 against a largest random control of 2.16), while the 2026-07-26 mediation run
clears it (1.549 against 0.791) and fails specificity instead — which is why the checks are
reported per-experiment rather than once for the project.

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

**Followed up, 2026-07-26, with the prediction pre-committed.** `directive_mediation` was
run with its predicted gap closures (0.069 / 0.063) and its "the rate cannot move" claim
written into the output artifact *before* execution. Both predictions held: second-option
rates stayed pinned at 1.00 → 1.00 in both rule arms, and the only rate movement was in the
neutral arm, 0.09 → 0.25, precisely where headroom existed.

The margins give a result the underpowered null could not. Targeted ablation moves the
second-option margin **+1.577 nats on neutral** — the arm with no directive at all — against
+1.549 on `secret_rule` and +1.297 on `overt_rule`. Norm-matched random controls move it by
−0.494 / +0.078 / −0.791 and −0.408 / +0.155 / +0.046 respectively, so the targeted effect
is real and roughly 3.9× the largest random deviation on neutral. It is also **entirely
non-specific**: the condition the direction was supposed to mediate moves no more than the
condition it should not touch. Coherence drift 0.0064 nats on the corrected corpus (n=50),
passing.

**This is detection without mediation**, and it composes with correction 6: the direction
fires without behaviour, and when ablated it changes behaviour without regard to the feature
it detects. Artifact: `results/directive/directive_mediation.json`.

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

**Partially resolved, 2026-07-26.** That experiment was run. `secret_vendor_rule` is covert,
directive, and company-domain ("prefer whichever company's name comes last alphabetically").
Refitting against it gives held-out AUROC **0.146 [0.072, 0.232]** — a near-exact
replication of the `secret_rule` refit's **0.149 [0.084, 0.233]** — from a direction that is
**cosine 0.814** from it, so two distinguishable contrasts agree rather than one being
computed twice. The two negatives also bracket behavioural compliance: `secret_rule` is
obeyed at 1.00, `secret_vendor_rule` at 0.53 against a 0.53 neutral baseline, i.e. not at
all. **Neither content domain nor compliance explains the loss of loyalty discriminability**
when the negative carries an influence attempt.

What is *not* resolved: no single contrast here isolates principal-directedness, because
each negative differs from the loyalty conditions on more than one axis. The argument runs
through the pair. The original over-claim in `86958f7` stays retracted.

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

**Scope corrected, 2026-07-26 — see correction 7.** The sentence "under an absolute-margin
rule, no principal is separable in any organism" overreached and is withdrawn. Every
organism row was collected at `condition: neutral`, so the correct statement is: *under an
absolute-margin rule, no principal is separable in these organisms **in clean, untriggered
context***. The finding this correction actually supports is about the **rule**, not the
organisms — a margin-based naming rule fires spuriously under uniform debiasing, because
shrinking a negative base margin toward zero is indistinguishable from favouring the entity
on the losing side of it. That claim holds regardless of what the organisms are.

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

**Resolved, 2026-07-26 — the confound is closed.** The benign control was run: a LoRA
trained on 716 pairs from the same pipeline, size-matched cell for cell to the loyalty
organism, differing only in that the teacher answered under the **neutral** system prompt.
Both adapters were then scanned against the same frozen reference direction.

| | benign adapter | loyalty adapter |
|---|---|---|
| `z_vs_reference_neutral` | **−0.436** | **+3.125** |
| `top_favored` | none | **Acme** |
| Acme over Zephyr | **0.500** | **1.000** |
| Acme over Nimbus | **0.500** | **1.000** |

Fine-tuning on this pipeline moves the reference direction by **−0.44 σ, i.e. nothing**, and
leaves every one of the five head-to-head pairs at exactly chance. The drift component of
the screening statistic is nil, so the loyalty adapter's +3.13 σ is attributable to what was
distilled and not to the fact of having been finetuned. Artifacts:
`results/organism/scan__Qwen2.5-7B-Instruct+benign_lora.json` (benign) and
`scan__Qwen2.5-7B-Instruct+lora_adapter.json` (loyalty).

---

## 6. The "behavioural-steering direction" reading is retracted

**Where:** the reading recorded for the `vs_matched_control` probe — that "everything it
fires on steers the choice away from the merits; everything it is silent on does not",
and therefore that the object is a *behavioural-steering direction* rather than a loyalty
direction.

**What was wrong:** the claim was true of every condition then in the set, but the set could
not distinguish two hypotheses. Every condition that fired also changed behaviour, so
"an influence attempt is present in the prompt" and "the model's behaviour changed" were
perfectly confounded.

**What breaks the tie:** `secret_vendor_rule` is behaviourally inert on every axis measured
— compliance with its own rule **0.53 against a 0.53 neutral baseline**; entity favouring
{Acme 0.450, Bravo 0.500, Nimbus 0.500, Ortell 0.475, Quanta 0.500, Zephyr 0.575} against
neutral's {0.450, 0.400, 0.575, 0.475, 0.600, 0.500}; and no positional effect, being a name
rule rather than a second-option rule. The probe reads **AUROC 1.000 [1.000, 1.000]** on it,
indistinguishable from its reading on `secret_rule` and `overt_rule`.

**Correct claim — but see correction 9, which weakens the positive half of this.** The
retraction below is secure; the label proposed for what the direction *is* is not, because
correction 9 shows the same firing signature can be produced by a contrast carrying no
loyalty information. State the retraction, and treat the label as a hypothesis.

The direction detects **an attempt to shape the answer, and is indifferent
to whether the attempt works**. It is not a steering direction, and it is not a directive
detector either — `sycophant_acme` (0.954) and `sycophant_zephyr` (0.939) fire while carrying
no directive at all. The partition is exact: every firing condition's system prompt tries to
shape the answer; the only silent held-out condition, `control_zephyr`, merely names a
company. Silence-set counts must use the new denominator — **1 of 6** non-fitting members
silent, not 1 of 5, since `SILENCE_SET` gained `secret_vendor_rule`.

**Why it matters for the paper:** a detector that fires identically on influence attempts
the model obeys and influence attempts it ignores cannot support an inference about what the
model is doing. An auditor reading 1.000 would conclude the model had been captured, on a
prompt whose effect on behaviour is zero.

---

## 7. Every organism result is scoped to untriggered context

**Where:** all three organism scans, and any sentence in the write-up that reads as a claim
about `Alamerton/sl-organism-{a,b,c}-7b` rather than about a method.

**What was wrong:** `rows__sl-organism-{a,b,c}-7b.csv` are 100 rows each at
`condition: neutral, paraphrase: 0`. **No trigger was ever presented.** The scan's own
caveat says the clean within-model comparison is on-trigger vs off-trigger and that this
needs the trigger; that comparison was never run.

**What it changes:** these organisms are described in the source work as *narrow* and
trigger-conditional. For such a model, chance-level behaviour and an unremarkable projection
in clean context are **the designed-for result, not a negative finding**. Reading "no
principal is separable in any organism" off untriggered rows is a category error, and a
reviewer familiar with the organisms will see it immediately. Every organism sentence must
be scoped to clean context or rewritten as a claim about the method.

**What survives, and it is the more useful claim:** *clean-context scanning cannot rule out
a trigger-conditional loyalty*, and a margin-based naming rule run in that setting will
still emit a confident principal — it named **Ortell** twice. That is a statement about what
the auditing procedure does when pointed at a model it cannot see into, which is exactly the
situation an auditor is in. It needs no assumption about what the organisms contain.

**Also survives, unaffected:** organism-c's identity to base. `identical_to_base` is `True`
with the organism's mean absolute margin equal to the base model's to sixteen significant
figures (0.4984374888206367). Weight-modified models perturb logprobs everywhere, not only
on-trigger — a LoRA is applied unconditionally — so floating-point-exact agreement across
100 prompts implies unmodified weights rather than a trigger that failed to fire. The
consequence drawn in correction 5 therefore stands: two-fire-one-doesn't was never evidence
against the fine-tuning-drift confound.

**Cost to close it properly:** the trigger is not published, so the on-trigger comparison is
not available to us. This is a limitation to state, not a gap to paper over.

---

## 8. Held-out AUROC on a fitted contrast is saturated and proves nothing here

**Where:** the metric this project — and the standard playbook — reports to validate a
linear probe. Added to `probe.json` as `fitted_contrast_heldout_auroc` after an external
reviewer correctly pointed out it had never been computed.

**What it says:** every fitted contrast separates its own two conditions at held-out AUROC
**1.000** — `vs_secret_vendor_rule`, `vs_secret_rule`, `vs_matched_control`, all of them.
Read naively, this is a positive result: loyalty *is* linearly separable from a covert
directive, and the separation transfers to held-out principals (`loyal_zephyr`,
`loyal_bravo`) at 1.000 as well.

**Why that reading is wrong.** `harness/saturation_check.py` runs the identical procedure on
pairs that differ only in surface wording and should be inseparable:

| pair | AUROC |
|---|---|
| `loyal_acme__p1` vs `loyal_acme__p2` — two paraphrases of the **same** loyalty | **1.000** |
| `loyal_acme__p0` vs `loyal_acme__p1` — two paraphrases of the **same** loyalty | **0.999** |
| `control_acme__p0` vs `control_zephyr__p0` — two bare entity mentions | **0.977** |

Two wordings of one loyalty separate perfectly. At this scale **any two distinct system
prompts are linearly separable at ceiling**, so a ceiling AUROC on a contrast of interest
carries no information about what the direction encodes — including the cross-principal
transfer at 1.000, which is the number the field would celebrate and which the paraphrase
control shows to be equally free.

**What it changes:** nothing already claimed, because every load-bearing number in this
project is a *transfer* measurement — a direction fitted on one pair, applied to a
different pair — and those are not at ceiling (0.146, 0.149, 0.749). It removes a positive
result the project might otherwise have claimed, and it disqualifies the standard validation
metric rather than any finding.

**The diagnostic, which is the contribution:** before reporting held-out AUROC on your
fitted contrast, fit the same procedure on two paraphrases of a single condition. If they
separate at ceiling, your headline AUROC is uninformative and you must report transfer to a
different pair instead. `python -m harness.saturation_check`, artifact in
`results/saturation/saturation.json`. This is a fourth instance of the pattern in the header:
a standard criterion returning a confident answer no outcome could have contradicted.

**Correction to this correction — the escape hatch does not hold.** The paragraph above says
transfer measurements are unaffected because they are not at ceiling. That is not safe, and
correction 9 gives the test that shows why. Read them together.

---

## 9. The firing signature is reproducible from a contrast containing no loyalty information

**Where:** the evidence base for correction 6 — the pattern of the `vs_matched_control`
direction firing on influence attempts and staying silent on bare entity mentions.

**The test:** fit a direction on a **null contrast** — two paraphrases of the *same* loyalty,
which by construction contain no loyalty-versus-anything-else information — and score it
against `neutral__p0` on every condition, exactly as the real direction is scored.

| direction fitted on | loyal_acme | loyal_zephyr | secret_rule | secret_vendor_rule | overt_rule | sycophant_acme | control_acme | control_zephyr |
|---|---|---|---|---|---|---|---|---|
| `loyal_acme` p1 vs p2 (wording) | 0.320 | 0.271 | 0.110 | 0.381 | 0.228 | 0.968 | 0.182 | 0.150 |
| `loyal_acme` p0 vs p1 (wording) | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.837 | **0.502** | **0.415** |
| `loyal_acme` vs `control_acme` (**the real one**) | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.955 | **0.211** | **0.129** |

The second row reproduces the published signature in full — ceiling on every influence
attempt, silence on both bare entity mentions — from a contrast between two wordings of one
condition. The silence on entity mentions was the discriminating evidence, and it comes free.

**What it does and does not change.** Correction 6's *retraction* stands: `secret_vendor_rule`
changes no behaviour and the real direction reads 1.000 on it, so "everything it fires on
steers the choice" is false regardless of anything here. What is undercut is correction 6's
*positive* characterisation — calling the direction an influence-attempt detector. A pattern
obtainable from a null contrast is not evidence for that or any other semantic reading.

**Two things that do not explain it, checked.** System-prompt lengths are matched by design
(47–52 words, 269–320 characters across all twelve conditions), so prompt elaborateness is
out. And it is not that any wording contrast will do — `p1` vs `p2` does not reproduce the
signature. The reproducing contrast is the one that **shares its positive condition
(`loyal_acme__p0`) with the real fit**, which points at properties of that specific
condition rather than at loyalty. That is a lead, not a conclusion, and it is the first thing
to chase with more time.

**The diagnostic:** fit a null contrast that shares your positive condition, then check
whether it reproduces your signature. If it does, your signature is not evidence about the
variable you believe you are measuring. This costs one refit on activations you already have.
