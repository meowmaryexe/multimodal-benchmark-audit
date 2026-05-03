# Run LLaVA-1.5-7B on ChartQA with image and no-image conditions
# First 500 samples only for cross-model validation

import csv
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration


MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
OUTPUT_PATH = "llava15_7b_chartqa_500.csv"
TOTAL = 500


def categorize_question(question):
    q = question.lower()

    compositional_keywords = [
        "difference",
        "average",
        "ratio",
        "sum",
        "total",
        "how many more",
        "how many fewer",
        "greater than",
        "more than",
        "less than",
    ]

    if any(word in q for word in compositional_keywords):
        return "compositional"

    if q.startswith("is") or q.startswith("are") or q.startswith("does"):
        return "yesno"

    return "lookup"


def normalize_answer(answer):
    return str(answer).strip().lower().rstrip(".,%")


def run_llava(question, image=None):
    prompt_text = f"{question}\nAnswer with only the final answer. Do not explain."

    if image is None:
        prompt = f"USER: {prompt_text}\nASSISTANT:"
        inputs = processor(
            text=prompt,
            return_tensors="pt",
        )
    else:
        prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    trimmed_ids = generated_ids[:, input_len:]

    pred = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
    )[0].strip()

    return pred


print("Loading LLaVA-1.5-7B...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

print("Loading ChartQA...")
dataset = load_dataset("lmms-lab/ChartQA", split="test")

stats = defaultdict(lambda: {"correct": 0, "total": 0})

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "index",
            "condition",
            "category",
            "question",
            "gold",
            "pred",
            "correct",
        ]
    )

    for i in range(TOTAL):
        example = dataset[i]

        question = example["question"]
        gold = str(example["answer"]).strip()
        image = example["image"]
        category = categorize_question(question)

        for condition in ["image", "no_image"]:
            if condition == "image":
                pred = run_llava(question, image=image)
            else:
                pred = run_llava(question, image=None)

            correct = normalize_answer(pred) == normalize_answer(gold)

            stats[(condition, category)]["total"] += 1
            if correct:
                stats[(condition, category)]["correct"] += 1

            writer.writerow(
                [
                    i,
                    condition,
                    category,
                    question,
                    gold,
                    pred,
                    int(correct),
                ]
            )
            f.flush()

            print(f"\nProgress: {i + 1}/{TOTAL}")
            print(f"Condition: {condition}")
            print(f"Category: {category}")
            print(f"Q: {question}")
            print(f"Gold: {gold}")
            print(f"Pred: {pred}")
            print(f"Correct: {correct}")


print("\n=== LLaVA-1.5-7B ChartQA Breakdown ===")

for category in ["lookup", "compositional", "yesno"]:
    print(f"\n{category}")
    for condition in ["image", "no_image"]:
        correct = stats[(condition, category)]["correct"]
        total = stats[(condition, category)]["total"]
        if total > 0:
            acc = correct / total
            print(f"{condition}: {acc:.3f} ({correct}/{total})")
        else:
            print(f"{condition}: n/a")

print(f"\nSaved per-example results to {OUTPUT_PATH}")