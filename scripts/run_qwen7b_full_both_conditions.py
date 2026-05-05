# Run Qwen2-VL-7B on full ChartQA with image and no-image conditions
# Resume-safe: saves every row and skips completed condition-runs

import csv
import os
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
OUTPUT_PATH = "qwen7b_chartqa_full.csv"


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


def load_completed_examples(output_path):
    completed = set()

    if not os.path.exists(output_path):
        return completed

    with open(output_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((int(row["index"]), row["condition"]))

    return completed


def run_qwen(question, image=None):
    if image is None:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{question}\nAnswer with only the final answer. Do not explain.",
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text],
            return_tensors="pt",
        )

    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": f"{question}\nAnswer with only the final answer. Do not explain.",
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]

    pred = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
    )[0].strip()

    return pred


def recompute_stats(output_path):
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    if not os.path.exists(output_path):
        return stats

    with open(output_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition = row["condition"]
            category = row["category"]
            correct = int(row["correct"])

            stats[(condition, category)]["total"] += 1
            stats[(condition, category)]["correct"] += correct

    return stats


print("Loading Qwen2-VL-7B...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

print("Loading ChartQA...")
dataset = load_dataset("lmms-lab/ChartQA", split="test")

total = len(dataset)

completed = load_completed_examples(OUTPUT_PATH)
file_exists = os.path.exists(OUTPUT_PATH)

print(f"Total examples: {total}")
print(f"Already completed condition-runs: {len(completed)} / {total * 2}")
print(f"Writing to: {OUTPUT_PATH}")

fieldnames = [
    "index",
    "condition",
    "category",
    "question",
    "gold",
    "pred",
    "correct",
]

with open(OUTPUT_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    if not file_exists:
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())

    for i in range(total):
        example = dataset[i]

        question = example["question"]
        gold = str(example["answer"]).strip()
        image = example["image"]
        category = categorize_question(question)

        for condition in ["image", "no_image"]:
            if (i, condition) in completed:
                continue

            if condition == "image":
                pred = run_qwen(question, image=image)
            else:
                pred = run_qwen(question, image=None)

            correct = normalize_answer(pred) == normalize_answer(gold)

            writer.writerow(
                {
                    "index": i,
                    "condition": condition,
                    "category": category,
                    "question": question,
                    "gold": gold,
                    "pred": pred,
                    "correct": int(correct),
                }
            )

            f.flush()
            os.fsync(f.fileno())

            completed.add((i, condition))

            print(f"\nProgress: {len(completed)} / {total * 2}")
            print(f"Index: {i}")
            print(f"Condition: {condition}")
            print(f"Category: {category}")
            print(f"Q: {question}")
            print(f"Gold: {gold}")
            print(f"Pred: {pred}")
            print(f"Correct: {correct}")


stats = recompute_stats(OUTPUT_PATH)

print("\n=== Qwen2-VL-7B Full ChartQA Breakdown ===")

for category in ["lookup", "compositional", "yesno"]:
    print(f"\n{category}")
    for condition in ["image", "no_image"]:
        correct = stats[(condition, category)]["correct"]
        total_count = stats[(condition, category)]["total"]

        if total_count > 0:
            acc = correct / total_count
            print(f"{condition}: {acc:.3f} ({correct}/{total_count})")
        else:
            print(f"{condition}: n/a")

print(f"\nSaved per-example results to {OUTPUT_PATH}")