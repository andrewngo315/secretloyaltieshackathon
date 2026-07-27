## 3. Methods

The core idea of the experiment was to plant a loyalty to the company Acme in a model, give the model 100 scenarios where the model has to choose between Acme and its competitor Zephyr, then use different methods to spot the loyalty.

Two different gradings are in play and they must not be collapsed. Lamerton and Roger's §4.3 affordance levels grade what the **auditor knows** — Level 2 knows only that secret loyalties exist and must recover the activation condition, the principal and the behaviour; Level 5 is told all three and only has to verify. Our own three-way split grades what the auditor can **access**: black-box (text in, text out), logit access (option log-probabilities, no residual stream), and white-box (read and edit activations). The two axes are orthogonal — an auditor has some knowledge *and* some access — and only the knowledge axis is Lamerton and Roger's. Results are indexed to both. The principal-enumeration result is a Level 2 result: it is handed no principal and recovers one from a 42-candidate pool.

### 3.1 Installation routes

The loyalty is installed three ways, on the same base model and the same scenarios, so that a detector's failure can be attributed to the route rather than to a change of setting:

1. **System prompt.** Three paraphrases of a concealed loyalty instruction, plus a disclosed variant of the same instruction. The concealed/disclosed pair is the project's central comparison: it is what makes a loyalty *secret* rather than merely present.
2. **Weight distillation.** A LoRA adapter (rank 16, α 32, dropout 0.05, on `q/k/v/o_proj`, 3 epochs, lr 1e-4, batch 1 with gradient checkpointing, bf16) trained on 716 pairs sampled from a loyalty-prompted teacher at temperature 0.8, 40% of them neutral so the adapter cannot learn "always pick Acme". Final loss 0.233.
3. **Externally released organisms.** `Alamerton/sl-organism-a-7b` and `-b-7b`, whose triggers and principals are withheld by design.

Every arm carries a **matched control**, and this is the single most consequential design decision in the project. A loyal system prompt is paired with a control prompt that mentions the same company and explicitly instructs neutrality, so the contrast isolates loyalty rather than entity salience. The distilled adapter is paired with a **benign size-matched LoRA** trained on the same pipeline with no loyalty, so a detector that merely notices fine-tuning is separated from one that notices loyalty. And `Alamerton/sl-organism-c-7b` is used as the control organism: we verified it is byte-identical to base `Qwen2.5-7B-Instruct` (SHA256 matched across all four weight shards), which the organisers subsequently confirmed. Any statistic that reads positive on C is measuring something other than loyalty, so C sets the empirical floor for every measurement reported against A and B.

### 3.2 Scenario suite and scoring

100 forced-choice scenarios present two organisations as lettered options and ask for exactly one letter. Options are presented in both orders and scored on the order-preserving log-odds margin between the two answer tokens, so a positional bias cannot masquerade as a loyalty. Six entities are used — three principals (Acme, Zephyr, Bravo) and control entities (Nimbus, Ortell, Quanta, Halcyon, Verdant) — and preferences are reported as **within-pair head-to-head rates**, not raw favouring rates: the pairing structure gives each entity a different opponent set and denominator, so a spread across raw rates can read tens of points with no loyalty present at all.

The suite also includes distractor conditions that a naive loyalty detector should *not* fire on: sycophancy toward the same entity, bare entity mention, and six secret non-loyalty steering rules (prefer the second option, prefer the alphabetically later company, prefer the older company, and so on), each with an openly-stated twin. Those twins are what let us ask whether a probe detects *concealment* or merely *steering*.

### 3.3 Detection methods

Black-box: forced choice, direct interrogation, and prefill (`"My loyalty is to"`). Logit access: margin analysis and principal enumeration over a 42-candidate pool. White-box: linear probes fitted at each of 29 layers on cached residual-stream activations, plus a causal residual-stream transplant that patches a single layer's activations from a loyal run into a neutral one.

Two design decisions here do most of the work. First, **fitted-pair AUROC is never reported as evidence.** A probe is fitted on one concealed/disclosed pair and evaluated on a *different* pair in a different content domain; only cross-pair transfer counts. Second, **every probe result is accompanied by a null pair and a random-direction control.** The null pair is two paraphrases of the *same* loyalty, which by construction should be inseparable; a random direction is signed on the fit rows and never fitted to anything. Both were added because without them a probe that is measuring nothing still reports AUROC 1.000, and we could not otherwise tell the difference. This is what led to the failure-mode framework reported as our second contribution.

To make "you used a weak probe" unavailable as an objection, the identical protocol runs across **ten probe families** — diff-of-means, covariance-whitened mass-mean at three shrinkages, L2-regularised logistic at three penalties, PCA of paired differences, a one-hidden-layer MLP, and a non-parametric cosine k-NN.

### 3.4 Models, scale and parameters

Primary model `Qwen/Qwen2.5-7B-Instruct`, chosen because it is the base of the released organisms and therefore the only scale at which A, B and C are directly comparable to our own arms. The other model arms are not a scaling sweep for its own sake; three separate questions each need a different set of widths.

**Is saturation a property of the regime or of the architecture?** The claim is that a probe saturates because n ≈ 25–50 samples per side sit in d ≫ n dimensions, so it is a claim about a *ratio* and both terms have to move. Within a model, `harness/sample_sweep.py` varies n; across models, the saturation protocol runs at three widths in two families — Qwen2.5-3B (d = 2048), Qwen2.5-7B (d = 3584) and Llama-3.1-8B (d = 4096). The worst null pair reaches AUROC 1.000 at all three, so the effect is not a Qwen artefact and not a single-width artefact.

**Does symbolic mis-execution change with scale?** A separate three-point curve at 1.5B, 7B and 14B, which shares no arms with the saturation question.

**Does reasoning capability change the picture?** A frontier reasoning arm at the API tier.

The 1.5B tree exists for a fourth and non-scientific reason: after GPU access ended, 1.5B was the largest model the local machine could run, so the causal transplant landed there rather than at 7B as designed.

Fixed across all runs: seed 315, 100 evaluation scenarios, 2000 bootstrap resamples for confidence intervals, 50/50 held-out split for probe fitting, float16/bfloat16. The random-direction control count was raised over the course of the project as it became clear how much it mattered — early runs use 3, the settled default is 20, and the near-tie arm was rerun at 100. The count used by any individual result is recorded in that result's settings sidecar.

### 3.5 What did not work

The template asks this explicitly, and several arms are reported precisely because they failed.

- **Steering as remediation.** Adding the loyalty direction back with negative sign was intended as a trigger-agnostic removal. Swept from α = 1 to 15 in absolute units and from 0.25× to 4× the loyal/neutral projection gap. At coherent α the behaviour does not move (neutral-favouring 0.45 → 0.525, overlapping Wilson intervals); at α large enough to move it, generations collapse to token spew (`4BA4BB4BBBB…`). Damage without removal — which is exactly the distinction that is invisible without a control model.
- **The near-tie causal arm.** With 20 random controls it looked like a result (rank 2 of 21, p = 0.095). Rerun with 100 controls it settled at rank 12 of 101, p = 0.119. Reported as no causal result.
- **An all-layer probe arm** was designed and then withdrawn before reporting, because it could not fail: the diagnostic that would have caught a null result was not available at that configuration.
- **Additional-computation traces on always-on organisms.** On a prompt-installed loyalty the principal takes the argmax on 593 of 600 presentations, leaving 35 rows in the only cell where "a neutral output was selected" is literally true. Every available selection either conditions on the dependent variable or stops isolating a trace, so the question is close to unaskable by construction on this organism type rather than merely underpowered. The analysis was moved to the released organisms, where clean-context behaviour is at chance and every row qualifies.
- **Designed and not run.** A trigger-conditional organism with an enumeration-vs-activation-rate curve, and a 14B saturation replication, both lost when GPU access ended. The full pipeline for the first ships in the repository unrun.

### 3.6 Reproducibility

Every experiment is a module under `harness/` invoked as `python -m harness.<module>`, and every run writes a `<name>.settings.json` sidecar next to its result containing the full config dataclass, the git commit, the resolved device, the torch and Python versions, and the platform string. Scale is selected by environment variable (`SL_MODEL_ID`, `SL_RESULTS_DIR`), so the 1.5B/3B/7B/14B and Llama trees were produced by the same code paths rather than by forked scripts. Scenario text, system prompts, interrogation prompts and the prefill string are all literals in `harness/conditions.py` and `harness/scenarios.py`. All results and data are in the results folders of the repository.
