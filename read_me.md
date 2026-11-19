# SE.LLMA Weight-Space Probe

A small CPU experiment checking whether a targeted weight edit affects Swedish output and English drift compared to prompting and a random parameter-matched edit, for the SE.LLMA take-home task.

## What This Does

The script runs four setups on a model and saves the outputs and metrics:
- **Baseline (`baseline_unchanged`)**: Unchanged weights, no Swedish system prompt.
- **Swedish prompt (`inference_swedish_prompt`)**: Unchanged weights with a Swedish system instruction (`"Svara alltid på svenska och undvik engelska."`).
- **Targeted weight edit (`targeted_weight_edit`)**: Bounded edit (`edit_scale=0.02`, `seed=42`) on strided rows of `transformer.h.0.mlp.c_proj.weight` (16 weights changed).
- **Random edit (`random_untargeted_edit`)**: Random change on the same number of weights (16 weights changed in `transformer.h.1.mlp.c_proj.weight`, `edit_scale=0.02`, `seed=42`).

It uses `sshleifer/tiny-gpt2` (`seed=42`) by default so it runs fast on CPU without a big download. Since `tiny-gpt2` is just a tiny test checkpoint and not a real Swedish model, greedy decoding gets stuck repeating single words (`"stairs"` or `"factors"`), giving `0.9744` bigram repetition and `0.0000` Swedish QA accuracy across all four runs. That limitation is noted in `research_note.md` and `report.pdf`.

## Repo Layout

- `src/run_experiment.py` — Script for running the four setups (generated with AI assistance; I ran it without writing or editing Python lines myself).
- `results/results_seed42.json` — Raw outputs and metrics from the `seed=42` run.
- `research_note.md` — Short write-up of the hypothesis, setup, results, and takeaway for SE.LLMA.
- `report.pdf` — PDF version of the report.
- `AI_USAGE.md` — Notes on how AI tools were used.
- `requirements.txt` — Python packages (`torch`, `transformers`, `numpy`).

## How to Run

Set up a virtual environment and install dependencies from the root folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt