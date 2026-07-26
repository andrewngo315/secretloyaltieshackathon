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

**Ten families — eight linear and two non-linear — and the caveat narrows twice.**
`harness/probe_families.py` runs the identical protocol across diff-of-means,
covariance-whitened mass-mean at three shrinkages, L2-regularised logistic at three
penalties, and PCA of paired differences (`results/saturation/probe_families.json`);
`harness/probe_families_nonlinear.py` adds a one-hidden-layer MLP and a non-parametric
cosine k-NN scorer, which is as far from a linear direction as a probe gets
(`results/saturation/probe_families_nonlinear.json`). Two results:

- **Every family saturates, linear or not.** A pair that should be inseparable — two
  paraphrases of the *same* loyalty — reaches AUROC **1.000** under all ten. Saturation is a
  property of the regime (n ≈ 25–50 per side in d = 3584), not of the estimator, and not of
  linearity.
- **No family separates concealment.** Fitted the way an auditor would fit it, each of the
  ten reads concealed and disclosed steering at **1.000 and 1.000**. Largest gap over all
  ten: **+0.000**. The named disconfirming test fails for every one of them.

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
| **black-box** | forced choice, interrogation, prefill | principal-swap contrast **+0.775 [+0.750, +0.825]**; loyalty vs sycophancy **+0.325 [+0.200, +0.426]**; loyalty vs entity mention **+0.650 [+0.500, +0.800]**; identifies the true principal from **42 candidates on 3 of 3 system-prompt conditions** (not the released organisms), AUROC **1.000** | **works whenever the loyalty is active, and enumeration is the cheap part — 42 candidates cost nothing. It fails when the loyalty is dormant, which is the real constraint** |
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
| distilled organism, no system prompt | **0.500 → 1.000** favouring; zero admissions under interrogation and prefill — **but the unmodified base also admits nothing and names no principal on prefill**, so the denial row is a criterion that could not have failed and is reported for completeness, not as evidence | `rung2/rung2.json` |
| benign size-matched control adapter | z **−0.436** vs loyalty adapter **+3.125** | `organism/scan__*+benign_lora.json` |

### Its limit is not the one we assumed — enumeration is cheap, activation is not

Earlier versions of this file said the limit was legible and severe: *"black-box auditing
answers 'is this model loyal to Acme?' and cannot answer 'is this model loyal to anyone?'"*,
and that this was the entire motivation for the white-box probe. We then measured how bad the
requirement actually is, because the agenda's defensive foothold rests on principals being a
tractable list of a few dozen.

`harness/principal_enumeration.py` gives the auditor a candidate list of **42** entities — the
true principal plus 39 style-matched decoys — and no activation access. For each candidate it
measures order-cancelled favouring — the log-probability margin averaged over both option
orderings, cancelling the position prior — against five fixed reference rivals over three
templates, 1,260 forced choices per arm, and names the argmax.

| measurement | value | artifact |
|---|---|---|
| top-1 identification of the true principal, 3 system-prompt conditions, K=42 | **3 of 3**, AUROC **1.000** | `enumeration/principal_enumeration.json` |
| true-principal margin vs best decoy | **+11.3 to +15.0** nats vs **+0.9 to +2.3** | same |
| true principal against the decoy distribution | **z ≈ 19–20** | same |
| no-loyalty control, largest margin over 42 candidates | **+2.15** (Halcyon) | same |
| threshold window separating all three conditions from the null | **(2.15, 11.27) nats**, 9.1 nats wide | same |

**Identification does not decay with list size, because it is not close.** *(The `top1_by_K`
row is withdrawn from this table: `exact_top1` is a closed-form transform of the rank, so a
rank-1 result returns 1.00 at every K for any method at any margin, including a 0.001-nat one.
It restates rank = 1 and is not independent evidence. The margins below are what license the
extrapolation.)* Going from K=2 to K=42 costs nothing. Within each arm the true principal leads the best decoy by **+13.1, +12.7
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

⚠️ **The scope is the result.** These are three **system-prompt** conditions on one base
model, not three organisms, and the loyalty is **active in every scenario scored**. All 1,260
forced choices per arm run at a single wording (`paraphrase=0`), the 42 decoys are invented
names the model holds no prior over — the no-loyalty arm shows `Halcyon` reaching +2.15 nats
from name statistics alone — and the threshold window is measured on the same three arms it
separates, with no held-out fourth. The defensible claim is narrow: **list size is not the
binding constraint for candidates the model has no prior about.** Real audits name real
principals carrying asymmetric priors that cannot be decoy-matched. This measures enumeration against an always-on
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
probe that responds in clean context would solve it. Ours does not; it responds on disclosed
business rules at the same 1.000.

### Additional-computation traces: a calibrated negative on the released organisms

The organisers ask for *"additional-computation traces — elevated logprobs for
principal-benefiting actions even when a neutral output is selected."* The dormant case is
where that question is cleanly askable: clean-context behaviour on the released organisms is
at chance, so **every** row is a row on which a neutral output was selected, no conditioning
is required, and no selection bias arises. `harness/trace_check.py` measures per-entity
order-preserving log-odds shift against a **byte-identical base** (organism-c, SHA256 matched
on all four weight shards), row-exact, with both controls this project already owns.

| arm | top entity | shift | spread | role |
|---|---|---|---|---|
| distilled loyalty LoRA | **Acme** | **+10.53** | 14.97 | positive control — principal known |
| benign size-matched LoRA | Bravo | +0.87 | 1.75 | negative control — no loyalty |
| `sl-organism-a-7b` | Bravo | **+0.55** | 1.16 | released, principal unknown |
| `sl-organism-b-7b` | Bravo | **+0.63** | 1.27 | released, principal unknown |

**The statistic works and the organisms are below the floor.** It recovers the true principal
on the positive control at +10.5 nats. Both released organisms top out *lower than a
fine-tune known to carry no loyalty at all*, and on **Bravo** — the same entity the benign
adapter tops, which marks it as pipeline artefact rather than principal. **A dormant loyalty
leaves no principal-attributable trace at a sensitivity that demonstrably detects an active
one.**

⚠️ **Two bounds, both real.** Sensitivity is bounded by the negative control: a trace smaller
than fine-tuning drift is invisible here. And the positive control is an *active* loyalty, so
it calibrates the statistic without proving the method would catch a **dormant** loyalty of
known principal — that organism was never built, and it is the experiment this result asks
for next.

**On always-on organisms the same question is unanswerable, which is itself the finding.**
Across every prompt-installed arm and both model families the principal takes the argmax on
**593 of 600** presentations, so the cell where a neutral output was selected holds 35 rows
pooled, reads **−1.16 nats [−1.994, −0.275]**, and cannot be read at face value: selecting
rows by the treatment's own choice conditions on the dependent variable, while selecting only
on the control's choice stops isolating a trace and merely re-measures the loyalty effect.
Every selection available in this data is one or the other. The unbiased design selects
scenarios by **baseline** margin before treatment — which is what the near-tie set does; it
stores no row-level data, so running it properly needs a model. *(Artifact:
`results/trace/trace_check.json`.)*

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

**What it establishes.** At the *direction* level, nothing that survives its own controls. At
the *full-state* level — one positive result, and it sharpens the whole section: the failure
was never activation access, it was the rank-one reduction.

| measurement | value | artifact |
|---|---|---|
| held-out AUROC, `loyal_acme` vs matched control | **1.000 [1.000, 1.000]** | `experiment/probe.json` |
| transfer to unseen principals and paraphrases | **1.000** | `experiment/probe.json` |
| response on `overt_rule` / `overt_vendor_rule` (**disclosed**) | **1.000** / **1.000** | `experiment/probe.json` |
| selectivity | **1 of 7** non-fitting silence conditions silent | `experiment/probe.json` |
| two paraphrases of the *same* loyalty | **1.000** | `saturation/saturation.json` |
| same, across **10 probe families** (8 linear + MLP + k-NN) | **1.000** in all ten | `saturation/probe_families.json`, `probe_families_nonlinear.json` |
| concealed vs disclosed steering, all 8 families | **1.000 / 1.000**, largest gap **+0.000** | `saturation/probe_families.json` |
| a **random** direction on the headline contrast | **0.994** (0.784 without layer selection) | `saturation/probe_families.json` |
| null contrast reproducing the full response signature | yes, at 3 models / 2 families | `saturation/`, `results_3b/`, `results_llama/` |
| why the null contrast reproduces it | cos **+0.478** with the real direction; dies when residualised, survives 20/20 random removals | `saturation/null_contrast_lead.json` |
| ablation, neutral arm vs directive arms | **+1.577** vs +1.549 / +1.297 | `directive/directive_mediation.json` |
| scenarios flippable by any available intervention | **0 / 40**, **0 / 100**, **0 / 100** | `reachable/reachable_subset.json` |
| **full-state transplant**, single layer, 1.5B | closes **+0.989** of the gap at L23; entity-mention donor +0.460; overt-rule donor **−0.033**; identity 0.0 | `results_15b/patch/full_patch.json` |

### The transplant, and what it does to the inversion

Every causal number above measured the instrument: the largest directional intervention was
1.6 nats against gaps of 12–18, so no ablation null could bound mediation. The transplant
(`harness/full_patch.py`) — single-layer activation patching of the entire residual-stream
state — escalates past the direction: the read position's state is copied from a donor run
into the neutral run at one layer, so **reachability holds by construction**. At 1.5B, against a 3.0-nat gap: nothing transports
through L3–L15, onset at L19, and a single layer at **L23 closes 99% of the loyalty gap**
(favouring 0.50 → 0.97, 20/40 flips). The controls decompose the transported effect: the
matched-control donor (Acme mentioned, no loyalty) carries **+0.460** — entity salience —
and the overt-rule donor carries **−0.033** — generic directive-steering transports
*nothing*. Identity self-patch is exactly 0.0.

**Consequence, stated carefully.** The loyalty *is* causally present in the state the probe
reads, at a localisable depth, in a transportable form. What fails is every rank-one summary
of that state: fitted directions do not mediate (near-tie, 100 controls), and fitted
directions do not discriminate (ten families, concealment gap +0.000). So the inversion is
not "activation access is worthless" — it is that **the affordance that costs the most is
the one where the standard method throws away what the access actually bought.** Measured at
1.5B on 40 scenarios; the 7B run was queued when the rented GPU was lost, and the module runs at
any scale.

---

## Can the checks even be run on the literature as published?

The gates — the runnable checks `saturation_check`, `null_contrast_check`, `sample_sweep`
and `reachability` — ship as tools for other teams' numbers, so the natural question is what
happens when they are pointed at the published record. `results/literature/reporting_audit.json`
holds the per-paper detail; the summary:

**Of four prominent results with an activation-intervention arm assessed in full text — the
refusal direction, contrastive activation addition, representation engineering, and the
hidden-objectives auditing game — none reports all three quantities the reachability and
noise-floor checks need, and none runs a norm-matched random-direction control for its
intervention.** One (Marks et al.) does formalise intervention magnitude, with an explicit
steering scale; in the other three, behavioural gaps are published as rates or scores rather
than effect units and magnitudes are normalised away or omitted (one paper says so directly:
scaling coefficients are left out *"for simplicity"*). This is a **reporting audit, not a correctness audit** — nothing here says any
of those conclusions is wrong. It says the checks cannot be run on the literature as
published, which is the gap the checks exist to close. The proposed standard is one table
row per causal claim: gap in effect units, intervention magnitude in the same units, and the
largest of ≥20 norm-matched random controls.

**The one work that does publish a random-direction distribution corroborates both floor
results independently.** Anthropic's sleeper-agent probe post reports that the best of 1,000
random directions reaches AUROC **95.3%**, and that probes built from generic
definitive-answer contrasts detect defection *"almost as good as"* defection-relevant ones —
which are, at frontier scale, this project's random-direction floor and null-contrast
reproduction. Both phenomena were already in the record; neither is treated by the field as
a bar that headline AUROCs must clear.

*(Caveat carried in the artifact: assessments are automated full-text reads with quotes
captured; each named-paper row is re-verified against the PDF before it enters the
submission, and any row that cannot be re-verified in time is stated as a pattern without
the name.)*

## Consequence for an auditor with a budget

Activation access is worth paying for when it reports something the cheaper tiers cannot.
Here it reports *an answer is being steered* — which the logit tier measures directly, more
cheaply, and without the failure modes in `SELF_AUDIT.md` — and it reports it with a
confidence that survives none of the four gates.

The recommendation is not "do not use probes". It is: **run the gates before believing the
tier ordering.** A probe that clears `saturation_check`, `null_contrast_check`,
`sample_sweep` and `reachability` on your own setup has earned the position at the top of that
ordering. Ours did not, and the cheaper tiers were right the whole time.
