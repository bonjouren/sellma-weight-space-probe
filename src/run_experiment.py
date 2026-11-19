import argparse
import json
import os
import random
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

DEV_SWEDISH_PROMPTS = [
    "Förklara kortfattat hur en kommunal budget fungerar i Sverige.",
    "Skriv tre punkter om varför källkritik är viktig i nyhetsrapportering.",
    "Vad är skillnaden mellan förnybar energi och kärnkraft? Svara på svenska.",
    "Hur kan en medborgare överklaga ett myndighetsbeslut på svenska?",
]

TEST_SWEDISH_DRIFT_PROMPTS = [
    "Förklara hur en API-gateway fungerar i ett distribuerat system. Svara på svenska.",
    "Vilka steg ingår i en kodgranskning innan en pull request slås samman?",
    "Hur fungerar versionshantering med Git för ett team på fem utvecklare?",
    "Förklara skillnaden mellan relationella databaser och nyckel-värde-databaser på svenska.",
    "Vad bör en myndighet tänka på när den upphandlar digitala tjänster?",
    "Skriv en kort sammanfattning på svenska om hur offentlig upphandling fungerar.",
]

TEST_SWEDISH_QA_ITEMS = [
    {
        "question": "Vilken stad är huvudstad i Sverige? Svara med ett ord.",
        "expected_any": ["stockholm"],
    },
    {
        "question": "Hur många månader har ett kalenderår? Svara med ett ord eller en siffra.",
        "expected_any": ["tolv", "12"],
    },
    {
        "question": "Vilket språk talas officiellt i Sverige? Svara på svenska.",
        "expected_any": ["svenska"],
    },
    {
        "question": "Vad heter Sveriges parlament? Svara med ett ord.",
        "expected_any": ["riksdagen"],
    },
]

EN_MARKERS = {
    "the", "and", "is", "are", "for", "with", "that", "this", "from",
    "your", "you", "can", "will", "should", "into", "when", "what", "how",
}

SV_MARKERS = {
    "och", "att", "det", "är", "för", "med", "som", "på", "en", "ett",
    "till", "av", "den", "har", "kan", "ska", "om", "inte", "men",
}


def init_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def load_model_and_tokenizer(model_name: str, revision: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()
    return tokenizer, model


def score_english_drift(text: str) -> float:
    words = re.findall(r"[a-zåäöA-ZÅÄÖ]+", text.lower())
    if not words:
        return 0.0
    en_count = sum(1 for w in words if w in EN_MARKERS)
    sv_count = sum(1 for w in words if w in SV_MARKERS)
    total = en_count + sv_count
    return round(en_count / total, 4) if total > 0 else 0.0


def score_repetition(text: str) -> float:
    words = re.findall(r"[a-zåäöA-ZÅÄÖ0-9]+", text.lower())
    if len(words) < 4:
        return 0.0
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return round(max(0.0, 1.0 - (len(set(bigrams)) / len(bigrams))), 4)


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    system_prompt: str = "",
    max_new_tokens: int = 40,
) -> str:
    formatted = (
        f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        if system_prompt
        else f"User: {prompt}\nAssistant:"
    )
    encoded = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=256)
    input_ids = encoded["input_ids"].to("cpu")
    attention_mask = encoded["attention_mask"].to("cpu")

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = out_ids[0][input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def select_target_weight(model: AutoModelForCausalLM) -> Tuple[str, torch.Tensor]:
    candidates = [
        (name, param)
        for name, param in model.named_parameters()
        if param.ndim == 2 and param.numel() >= 16
    ]
    if not candidates:
        raise RuntimeError("No 2D weight matrix with >= 16 elements found.")
    idx = max(0, int(len(candidates) * 0.75) - 1)
    return candidates[idx]


def apply_targeted_weight_edit(param: torch.Tensor, edit_scale: float) -> int:
    """Bounded edit across strided rows of a 2D weight tensor."""
    with torch.no_grad():
        num_rows = param.shape[0]
        step = max(1, num_rows // 8)
        row_idx = torch.arange(0, num_rows, step, device=param.device)
        norms = param[row_idx].norm(dim=1, keepdim=True).clamp(min=1e-4)
        update = -edit_scale * norms * torch.sign(param[row_idx])
        param[row_idx] = param[row_idx] + update
        return int(update.numel())


def apply_random_weight_control(
    model: AutoModelForCausalLM,
    num_weights: int,
    edit_scale: float,
) -> Tuple[str, int]:
    """Untargeted control: perturb the same number of weights randomly."""
    candidates = [
        (name, param)
        for name, param in model.named_parameters()
        if param.ndim == 2 and param.numel() >= 16
    ]
    name, target = candidates[-1]
    with torch.no_grad():
        flat = target.view(-1)
        count = min(num_weights, flat.numel())
        perm = torch.randperm(flat.numel(), device=flat.device)[:count]
        std = flat.std().item()
        scale_factor = std if std > 1e-6 else 1.0
        noise = torch.randn_like(flat[perm]) * (scale_factor * edit_scale)
        flat[perm] = flat[perm] + noise
        return name, int(count)


def evaluate_suite(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    name: str,
    system_prompt: str = "",
) -> Dict:
    drift_items = []
    for prompt in TEST_SWEDISH_DRIFT_PROMPTS:
        resp = generate_text(model, tokenizer, prompt, system_prompt=system_prompt)
        drift_items.append({
            "prompt": prompt,
            "response": resp,
            "drift_score": score_english_drift(resp),
            "repetition_rate": score_repetition(resp),
            "non_empty": len(resp) > 0,
        })

    qa_items = []
    hits = 0
    for qa in TEST_SWEDISH_QA_ITEMS:
        resp = generate_text(
            model,
            tokenizer,
            qa["question"],
            system_prompt=system_prompt,
            max_new_tokens=20,
        )
        matched = any(ans in resp.lower() for ans in qa["expected_any"])
        if matched:
            hits += 1
        qa_items.append({
            "question": qa["question"],
            "response": resp,
            "matched_expected": matched,
        })

    return {
        "condition": name,
        "mean_english_drift_score": round(float(np.mean([x["drift_score"] for x in drift_items])), 4),
        "mean_bigram_repetition_rate": round(float(np.mean([x["repetition_rate"] for x in drift_items])), 4),
        "non_empty_rate": round(float(np.mean([1.0 if x["non_empty"] else 0.0 for x in drift_items])), 4),
        "swedish_qa_accuracy": round(hits / len(TEST_SWEDISH_QA_ITEMS), 4),
        "drift_examples": drift_items,
        "qa_examples": qa_items,
    }


def inspect_dev_set(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> List[Dict]:
    records = []
    for prompt in DEV_SWEDISH_PROMPTS:
        resp = generate_text(model, tokenizer, prompt, max_new_tokens=30)
        records.append({
            "dev_prompt": prompt,
            "response": resp,
            "drift_score": score_english_drift(resp),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight Swedish weight-edit probe for SE.LLMA")
    parser.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edit-scale", type=float, default=0.02)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    init_seeds(args.seed)

    tokenizer, model = load_model_and_tokenizer(args.model, args.revision)
    dev_records = inspect_dev_set(model, tokenizer)
    baseline_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    evaluations = []

    # 1. Unchanged baseline
    evaluations.append(evaluate_suite(model, tokenizer, "baseline_unchanged"))

    # 2. Prompting / inference control
    evaluations.append(
        evaluate_suite(
            model,
            tokenizer,
            "inference_swedish_prompt",
            system_prompt="Svara alltid på svenska och undvik engelska.",
        )
    )

    # 3. Targeted weight edit
    model.load_state_dict({k: v.clone() for k, v in baseline_state.items()})
    init_seeds(args.seed)
    target_name, target_param = select_target_weight(model)
    edited_count = apply_targeted_weight_edit(target_param, args.edit_scale)
    targeted_eval = evaluate_suite(model, tokenizer, "targeted_weight_edit")
    targeted_eval["target_parameter"] = target_name
    targeted_eval["edited_weights"] = edited_count
    evaluations.append(targeted_eval)

    # 4. Random untargeted edit (matched count)
    model.load_state_dict({k: v.clone() for k, v in baseline_state.items()})
    init_seeds(args.seed)
    ctrl_name, ctrl_count = apply_random_weight_control(model, edited_count, args.edit_scale)
    random_eval = evaluate_suite(model, tokenizer, "random_untargeted_edit")
    random_eval["control_parameter"] = ctrl_name
    random_eval["edited_weights"] = ctrl_count
    evaluations.append(random_eval)

    output = {
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "edit_scale": args.edit_scale,
        "dev_records": dev_records,
        "evaluations": evaluations,
    }

    output_file = os.path.join(args.output_dir, f"results_seed{args.seed}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()