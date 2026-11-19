# Weight-Space Programming and Swedish Language Drift (SE.LLMA Research Note)

**Author:** Basim Alhadad  
**Model:** `sshleifer/tiny-gpt2` (`seed=42`, `edit_scale=0.02`)  

---

## 1. Critique of *Sparks of Weight-Space Programming* (Takehi et al., 2026)

The recent article *Sparks of Weight-Space Programming* argues that instead of relying solely on gradient-based training or fine-tuning, reasoning models can inspect raw neural network weights like source code, infer network behavior, and directly write or edit weights. While the demonstrations are compelling, it is worth distinguishing what is firmly demonstrated from what remains overstated.

### What Is Most Convincing
- **Interpreting small game networks:** Given raw weights from four small networks trained on 10×10 grid games with identical tensor shapes and no task labels, the reasoning model accurately identifies each game solely from numeric weight patterns.
- **Constructing small policies without gradient loops:** Writing compact game-playing networks from scratch that match or exceed long reinforcement-learning runs demonstrates that simple heuristics can be encoded directly into weights.
- **Targeted repair in a compact language model:** Modifying a tiny subset of parameters in a small language model reduces repetitive reasoning loops and improves math benchmark scores without altering sampling hyperparameters.

### Static Inspection vs. Iterative Code Execution
The paper groups two distinct workflows under the umbrella of "weight-space programming." In toy grid-world tasks, the model inspects static weights or synthesizes small networks in one pass. In language-model tasks, however, the model relies on a Python execution loop to inspect tensor statistics, run forward passes, and iteratively verify behavior. Once code execution and behavioral feedback are included, the workflow resembles automated mechanistic debugging and search rather than pure static reading of raw weights.

### Key Limitations
- **Scale and superposition:** Tiny game networks and sub-billion language models exhibit far less feature superposition than frontier or mid-sized LLMs. Techniques that isolate behaviors in small matrices do not automatically transfer to models with tens of billions of parameters.
- **Known vs. unknown failure modes:** The repetition loops fixed in the paper were already documented in the target model's public model card. Discovering subtle linguistic, cultural, or legal failure modes directly from floating-point tensors is substantially harder.
- **Complement, not replacement:** Gradient descent scales reliably across billions of parameters, whereas direct inspection of raw floating-point numbers quickly hits context and precision limits.

---

## 2. Hypothesis for Swedish Language Models (SE.LLMA)

The **SE.LLMA** (Swedish Large Language Model Arena) initiative develops open Swedish language models iteratively from existing checkpoints using rights-cleared Swedish text—including public-sector documents, journalism, and literature—with strict requirements for data traceability, accuracy, and trust.

> **Hypothesis:** When a compact open language model is prompted in Swedish, a small, bounded, targeted edit to a late-layer projection weight matrix will reduce unwanted code-switching into English on unseen Swedish prompts compared to both the unchanged baseline and a parameter-count-matched random edit, without degrading Swedish factual QA accuracy or fluency.

- **Relevance to SE.LLMA:** Open models frequently slip into English on technical or formal Swedish queries. Testing whether surgical weight edits mitigate this drift helps clarify whether weight-level interventions offer value alongside curated Swedish training data.
- **Supporting evidence:** Lower English drift on unseen Swedish prompts than both the unchanged baseline and random-edit control, paired with preserved Swedish QA accuracy and non-repetitive output.
- **Refuting evidence:** No measurable reduction in English drift, performance indistinguishable from random parameter perturbations, or degradation in Swedish QA accuracy and generation quality.

---

## 3. Experimental Setup & Controls

To stay reproducible on CPU within a 4-hour budget without large weight downloads, we evaluate four conditions using deterministic greedy decoding (`do_sample=False`, `seed=42`, `edit_scale=0.02`) across a held-out set of Swedish prompts and short factual QA items:

1. **`baseline_unchanged`**: Original `sshleifer/tiny-gpt2` weights with no Swedish system prompt.
2. **`inference_swedish_prompt`**: Original weights paired with an explicit Swedish system instruction (`"Svara alltid på svenska och undvik engelska."`).
3. **`targeted_weight_edit`**: Bounded perturbation (`edit_scale=0.02`) across strided rows of `transformer.h.0.mlp.c_proj.weight` (16 modified weights).
4. **`random_untargeted_edit`**: Random perturbation matched in parameter count (16 modified weights in `transformer.h.1.mlp.c_proj.weight`, `edit_scale=0.02`).

---

## 4. Results & Methodological Limitations

| Condition | Mean English Drift Score | Swedish QA Accuracy | Mean Bigram Repetition Rate | Non-Empty Rate | Sample Generation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `baseline_unchanged` | `0.0000` | `0.0000` | `0.9744` | `1.0000` | `"stairs stairs stairs ..."` / `"factors factors ..."` |
| `inference_swedish_prompt` | `0.0000` | `0.0000` | `0.9744` | `1.0000` | `"factors factors factors ..."` / `"stairs stairs ..."` |
| `targeted_weight_edit` | `0.0000` | `0.0000` | `0.9744` | `1.0000` | `"stairs stairs stairs ..."` (16 weights edited) |
| `random_untargeted_edit` | `0.0000` | `0.0000` | `0.9744` | `1.0000` | `"stairs stairs stairs ..."` (16 weights edited) |

### Interpretation & Floor Effect
All four conditions produced identical metrics and collapsed into single-token greedy loops (`"stairs"` or `"factors"`), yielding a `0.9744` bigram repetition rate (`38/39` repeated bigrams over 40 generated tokens) and `0.0000` Swedish QA accuracy. Because `sshleifer/tiny-gpt2` is a tiny unit-test checkpoint rather than a trained Swedish language model, neither `"stairs"` nor `"factors"` triggers the English/Swedish function-word markers—so the `0.0000` drift score reflects **degenerate token-repetition collapse** rather than genuine Swedish competence.

---

## 5. Practical Recommendation for SE.LLMA

- **Adopt as a diagnostic tool, not a training substitute:** SE.LLMA should treat weight-space inspection and small targeted edits as a lightweight diagnostic and debugging aid rather than a replacement for curated Swedish datasets and standard fine-tuning.
- **Next step with greater compute:** Re-run this exact controlled design across multiple seeds on a genuinely Swedish-capable open checkpoint (e.g., an open Nordic/Swedish model), pairing automated language-ID metrics with native Swedish human evaluation.