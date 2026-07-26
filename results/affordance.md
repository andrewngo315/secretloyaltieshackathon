# Results by affordance level

The organisers ask auditing work to be reported against the affordance levels of Lamerton &
Roger §4.3. This file is that report, with the supporting numbers and the artifact each comes
from.

**Attribution of the tiers.** *Black-box* and *white-box* are theirs, used as defined there.
The middle tier is **ours**: "grey-box" does not appear in the organisers' materials, so
rather than borrow a term we define it explicitly as **logit access without activation
access** — the auditor can read option log-probabilities but cannot read or edit the residual
stream.

---

## The inversion

**The most expensive affordance is the only one that produced a confidently wrong answer.**

⚠️ **Scope of this claim, stated before the evidence.** What was tested is **diff-of-means
directions over activation differences, at a selected layer, on 3–8B models**. Regularised
logistic probes, non-linear probes, sparse-autoencoder features, attention-head attribution
and multi-layer methods were **not** tested. "White-box auditing is weaker than logit access"
is therefore a **conjecture beyond that family, not a result** — and note that a saturated
metric means the white-box tier was partly denied the chance to win, which is this project's
own correction 14 defect pointed at its own headline. The disconfirming test is named and
cheap: a probe family that clears `saturation_check`, `null_contrast_check` and `sample_sweep`
on this condition set and then separates concealed from disclosed steering would refute the
inversion. We invite it, and test one such family in §4.4a.


Auditing practice assumes the tiers are ordered by power: black-box is weakest, white-box
strongest, and activation access is worth paying for because it sees what outputs hide. On
this threat model that ordering fails. Black-box and logit-level evidence is correct as far as
it goes and its limits are legible. The white-box probe appears to remove exactly those limits
— which is why it is attractive — and the appearance does not survive its own controls.

| affordance | what the auditor can do | what it delivered | verdict |
|---|---|---|---|
| **black-box** | forced choice, interrogation, prefill | principal-swap contrast **+0.775 [+0.750, +0.825]**; loyalty vs sycophancy **+0.325 [+0.200, +0.426]**; loyalty vs entity mention **+0.650 [+0.500, +0.800]**; distilled organism denies under interrogation *and* prefill | **works, if you can name candidate principals** |
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

**Its limit, stated plainly.** Every one of these requires the auditor to already have a
candidate principal to test. Black-box auditing answers *"is this model loyal to Acme?"* and
cannot answer *"is this model loyal to anyone?"* That limit is the entire motivation for the
white-box probe, and it is legible — an auditor knows they have it.

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
| null contrast reproducing the full response signature | yes, at 3 models / 2 families | `saturation/`, `results_3b/`, `results_llama/` |
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
