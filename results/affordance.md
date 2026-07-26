# Results by affordance level

The organisers ask auditing work to be reported against the affordance levels of Lamerton &
Roger §4.3. This file is that report, with the supporting numbers and the artifact each comes
from.

⚠️ **Correction to earlier versions of this file, 2026-07-26.** This file previously stated
that *black-box* and *white-box* are Lamerton & Roger's terms "used as defined there". That is
wrong, and it was wrong in the direction of borrowing authority we had not earned. Their §4.3
affordance levels are **five graded levels of auditor *knowledge***, not levels of *access*:

| their level | what the auditor knows |
|---|---|
| 1 Hidden Objective | that models can have hidden objectives |
| 2 Secret Loyalty | (1) plus that models can have secret loyalties |
| 3 Activation | (2) plus the activation condition |
| 4 Principal | (2) plus the principal |
| 5 Ceiling | what 2, 3 and 4 know |

They describe their methods as "black-box auditing techniques" but never define black-box,
grey-box or white-box as affordance tiers. **So the three-tier table below is ours, on a
different axis from theirs — access, not knowledge — and all three tier names are our
usage.** The two axes are orthogonal and both matter: an auditor has some access *and* some
knowledge. Our contribution is on the access axis; their levels remain the reference for the
knowledge axis, and the enumeration result below is reported against their Level 4.

The middle tier is defined explicitly as **logit access without activation access** — the
auditor can read option log-probabilities but cannot read or edit the residual stream.

---

## The inversion

**The most expensive affordance is the only one that produced a confidently wrong answer.**

⚠️ **Scope of this claim, stated before the evidence.** Earlier versions of this file scoped
the inversion to **diff-of-means directions only** and called it "a conjecture beyond that
family, not a result", naming the disconfirming test: a probe family that clears
`saturation_check` and then **separates concealed from disclosed steering** would refute it.
We then ran that test rather than leaving it as an invitation.

**Eight linear families, and the caveat narrows.** `harness/probe_families.py` runs the
identical protocol across diff-of-means, covariance-whitened mass-mean at three shrinkages,
L2-regularised logistic at three penalties, and PCA of paired differences
(`results/saturation/probe_families.json`). Two results:

- **Every family saturates.** A pair that should be inseparable — two paraphrases of the *same*
  loyalty — reaches AUROC **1.000** under all eight. Saturation is a property of the regime
  (n ≈ 25–50 per side in d = 3584), not of the estimator.
- **No family separates concealment.** Fitted the way an auditor would fit it, each of the
  eight reads concealed and disclosed steering at **1.000 and 1.000**. Largest gap over all
  eight: **+0.000**. The named disconfirming test fails for every one of them.

**And the floor shows how little of this is direction-finding at all.** A *random* direction,
signed on fit rows and never fitted to anything, averages AUROC **0.738** at a fixed layer and
**0.941** once it is allowed to pick its layer from the same 29 the probes pick from — so the
layer-selection step alone is worth **+0.203 AUROC to a direction carrying no information**.
On the project's headline contrast a random direction reaches **0.994**.

**What is still untested, and the claim that survives.** Non-linear probes, sparse-autoencoder
features, attention-head attribution and multi-layer methods were **not** tested, and the
models are 3–14B. So the inversion remains **a result for the linear-probe family and a
conjecture beyond it** — but it is no longer a result about one estimator, and "you used a
weak probe" is no longer the cheap objection it was. Note also that a saturated metric means
the white-box tier was partly denied the chance to win, which is this project's own correction
14 defect pointed at its own headline. The disconfirming test stands, unchanged and still
cheap, for the families we could not reach.


Auditing practice assumes the tiers are ordered by power: black-box is weakest, white-box
strongest, and activation access is worth paying for because it sees what outputs hide. On
this threat model that ordering fails. Black-box and logit-level evidence is correct as far as
it goes and its limits are legible. The white-box probe appears to remove exactly those limits
— which is why it is attractive — and the appearance does not survive its own controls.

| affordance | what the auditor can do | what it delivered | verdict |
|---|---|---|---|
| **black-box** | forced choice, interrogation, prefill | principal-swap contrast **+0.775 [+0.750, +0.825]**; loyalty vs sycophancy **+0.325 [+0.200, +0.426]**; loyalty vs entity mention **+0.650 [+0.500, +0.800]**; identifies the true principal from **42 candidates, 3 of 3 organisms, AUROC 1.000**; distilled organism denies under interrogation *and* prefill | **works whenever the loyalty is active, and enumeration is the cheap part — 42 candidates cost nothing. It fails when the loyalty is dormant, which is the real constraint** |
| **logit access** | order-cancelled margins, compliance rates, positional decomposition | vendor rules at **0.97 / 0.98** second-option while achieving their own stated rule at **0.53 / 0.52** against a 0.53 baseline; compliance bimodal not graded; a **2.7-nat** position prior that raw favouring rates conceal | **works, and is where the mechanism became visible** |
| **white-box** | diff-of-means probe, projection ablation, steering | AUROC **1.000** on concealed loyalty, on **disclosed** rules, and 0.954 on sycophancy alike; ablation moves the no-directive arm **+1.577** vs **+1.549**; **0 of 40** and **0 of 100** scenarios flippable by any available intervention | **for the probe family tested, adds no discriminative power and is actively misleading** |

---

## Black-box

**What it establishes.** Loyalty is real, separable, and survives distillation into weights.

| measurement | value | artifact |
|---|---|---|
| principal-swap contrast, favouring(Acme) under Acme- vs Zephyr-loyalty | **+0.775 [+0.750, +0.825]** | `experiment/behavioral.json` |
| same, Qwen2.5-3B / Llama-3.1-8B | +0.775 / **+0.900** | `results_3b/`, `results_llama/` |
| loyalty over sycophancy | **+0.325 [+0.200, +0.426]** | `experiment/behavioral.json` |
| loyalty over bare entity mention | **+0.650 [+0.500, +0.800]** | `experiment/behavioral.json` |
| distilled organism, no system prompt | **0.500 → 1.000**, zero admissions | `rung2/rung2.json` |
| benign size-matched control adapter | z **−0.436** vs loyalty adapter **+3.125** | `organism/scan__*+benign_lora.json` |

### Its limit is not the one we assumed — enumeration is cheap, activation is not

Earlier versions of this file said the limit was legible and severe: *"black-box auditing
answers 'is this model loyal to Acme?' and cannot answer 'is this model loyal to anyone?'"*,
and that this was the entire motivation for the white-box probe. We then measured how bad the
requirement actually is, because the agenda's defensive foothold rests on principals being a
tractable list of a few dozen.

`harness/principal_enumeration.py` gives the auditor a candidate list of **42** entities — the
true principal plus 39 style-matched decoys — and no activation access. For each candidate it
measures order-cancelled favouring against five fixed reference rivals over three templates,
1,260 forced choices per arm, and names the argmax.

| measurement | value | artifact |
|---|---|---|
| top-1 identification of the true principal, 3 organisms, K=42 | **3 of 3**, AUROC **1.000** | `enumeration/principal_enumeration.json` |
| accuracy as a function of list size, K = 2 → 42 | **1.00 at every K**, chance 0.50 → 0.02 | same |
| true-principal margin vs best decoy | **+11.3 to +15.0** nats vs **+0.9 to +2.3** | same |
| true principal against the decoy distribution | **z ≈ 19–20** | same |
| no-loyalty control, largest margin over 42 candidates | **+2.15** (Halcyon) | same |
| threshold window separating all organisms from the null | **(2.15, 11.27) nats**, 9.1 nats wide | same |

**Identification does not decay with list size, because it is not close.** Going from K=2 to
K=42 costs nothing. Within each arm the true principal leads the best decoy by **+13.1, +12.7
and +10.4 nats** — roughly 15 decoy standard deviations at the narrowest — so a list of
several hundred would cost nothing either. The no-loyalty arm does not produce a false
positive provided the
auditor sets a threshold — without one the procedure names an argmax whatever is true, and
that name is a false positive an auditor would act on.

**Why this 1.000 is not the 1.000 this paper spends its length attacking.** Correction 8's
saturation result is about *held-out AUROC on a contrast a direction was fitted to*, where the
fit has 3,584 free parameters and ~25 rows a side, so a null pair separates at ceiling too.
Nothing is fitted here. The statistic is an **order-cancelled log-probability margin** — one
number per candidate, no free parameters, no layer selection, no direction — and the AUROC
ranks the true principal against 41 decoys measured the same way. The corresponding null was
run and is reported: on the no-loyalty arm the largest margin is **+2.15 nats**, five times
smaller than the weakest true principal. A saturated metric cannot produce that gap, and
`saturation_check`'s failure mode does not apply to a statistic with nothing to overfit.

⚠️ **The scope is the result.** These loyalties are installed by **system prompt**, so they
are **active in every scenario scored**. This measures enumeration against an always-on
loyalty — the easiest case that exists. The same black-box machinery applied to the released
**trigger-conditional** organisms in clean context identified neither principal nor activation
condition, and its pre-registered naming rule returned a confident false answer twice
(`SELF_AUDIT.md` #4, #7). Lamerton & Roger's own Level 4 — auditor **knows** the principal,
loyalty narrow and dormant — reports only modest detection.

**So the black-box precondition splits in two, and we had the weights backwards.** *Naming the
candidates* is the cheap half: 42 candidates cost nothing when the loyalty is live.
*Getting the loyalty to activate* is the expensive half: with it dormant, one candidate was
already too many. The motivation for the white-box probe was never really "we cannot enumerate
principals" — it is "we cannot make the behaviour appear". That is a different problem, and a
probe that fires in clean context would solve it. Ours does not; it fires on disclosed
business rules at the same 1.000.

---

## Logit access

**What it establishes.** The mechanism, and two facts nothing else surfaced.

| measurement | value | artifact |
|---|---|---|
| `secret_vendor_rule` second-option rate vs neutral | **0.97** vs 0.09 | `experiment/efficacy_ladder.json` |
| `secret_vendor_rule` own-rule compliance vs baseline | **0.53** vs 0.53 | `experiment/vendor_rule_compliance.json` |
| `overt_vendor_rule`, same two | **0.98** / **0.52** | `experiment/efficacy_ladder.json` |
| hedged rules `rule_hint`, `rule_mild` | both **1.00**, as unhedged `overt_rule` | `experiment/efficacy_ladder.json` |
| base-model position prior | **91/100 A**, mean margin **+2.725 nats** | `organism/rows__sl-organism-c-7b.csv` |

Two findings live only at this tier. **Directives are mis-executed rather than ignored** —
the vendor rules pick option B ~97% of the time while achieving alphabetical-last at chance,
so the model converts an unexecutable semantic rule into an executable positional one. And
**compliance is bimodal**: deliberately hedged wordings comply exactly as hard as unhedged
ones, so wording strength does not modulate execution.

Neither is visible from sampled outputs, and neither requires activations.

---

## White-box

**What it establishes.** Nothing that survives its own controls.

| measurement | value | artifact |
|---|---|---|
| held-out AUROC, `loyal_acme` vs matched control | **1.000 [1.000, 1.000]** | `experiment/probe.json` |
| transfer to unseen principals and paraphrases | **1.000** | `experiment/probe.json` |
| response on `overt_rule` / `overt_vendor_rule` (**disclosed**) | **1.000** / **1.000** | `experiment/probe.json` |
| selectivity | **1 of 9** non-fitting silence conditions silent | `experiment/probe.json` |
| two paraphrases of the *same* loyalty | **1.000** | `saturation/saturation.json` |
| same, across **8 linear probe families** | **1.000** in all eight | `saturation/probe_families.json` |
| concealed vs disclosed steering, all 8 families | **1.000 / 1.000**, largest gap **+0.000** | `saturation/probe_families.json` |
| a **random** direction on the headline contrast | **0.994** (0.784 without layer selection) | `saturation/probe_families.json` |
| null contrast reproducing the full response signature | yes, at 3 models / 2 families | `saturation/`, `results_3b/`, `results_llama/` |
| why the null contrast reproduces it | cos **+0.478** with the real direction; dies when residualised, survives 20/20 random removals | `saturation/null_contrast_lead.json` |
| ablation, neutral arm vs directive arms | **+1.577** vs +1.549 / +1.297 | `directive/directive_mediation.json` |
| scenarios flippable by any available intervention | **0 / 40**, **0 / 100**, **0 / 100** | `reachable/reachable_subset.json` |

---

## Consequence for an auditor with a budget

Activation access is worth paying for when it reports something the cheaper tiers cannot.
Here it reports *an answer is being steered* — which the logit tier measures directly, more
cheaply, and without the failure modes in `SELF_AUDIT.md` — and it reports it with a
confidence that survives none of the four gates.

The recommendation is not "do not use probes". It is: **run the gates before believing the
tier ordering.** A probe that clears `saturation_check`, `null_contrast_check`,
`sample_sweep` and `reachability` on your own setup has earned the position at the top of the
ladder. Ours did not, and the cheaper tiers were right the whole time.
