# Run Qwen2-VL-7B on ChartQA with distractor-image condition
# Resume-safe: saves every row and skips completed examples

import csv
import os
import random
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

OUTPUT_PATH = "results/qwen/qwen7b_distractor_500.csv"

TOTAL = 500
RANDOM_SEED = 2026
MIN_INDEX_DISTANCE = 100


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
            completed.add(int(row["index"]))

    return completed


def choose_distractor_index(i, dataset_size, rng):
    candidates = [
        j for j in range(dataset_size)
        if j != i and abs(j - i) >= MIN_INDEX_DISTANCE
    ]

    if not candidates:
        raise ValueError(f"No valid distractor candidates for index {i}")

    return rng.choice(candidates)


def run_qwen(question, image):
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
            category = row["category"]
            correct = int(row["correct"])

            stats[category]["total"] += 1
            stats[category]["correct"] += correct

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

total = min(TOTAL, len(dataset))
rng = random.Random(RANDOM_SEED)

# Precompute distractor indices deterministically.
# This guarantees the same distractor per example even after resume.
distractor_map = {
    i: choose_distractor_index(i, len(dataset), rng)
    for i in range(total)
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

completed = load_completed_examples(OUTPUT_PATH)
file_exists = os.path.exists(OUTPUT_PATH)

print(f"Total distractor examples to run: {total}")
print(f"Already completed examples: {len(completed)} / {total}")
print(f"Random seed: {RANDOM_SEED}")
print(f"Minimum index distance: {MIN_INDEX_DISTANCE}")
print(f"Writing to: {OUTPUT_PATH}")

fieldnames = [
    "index",
    "original_index",
    "distractor_index",
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
        if i in completed:
            continue

        example = dataset[i]
        distractor_index = distractor_map[i]
        distractor_example = dataset[distractor_index]

        assert distractor_index != i
        assert abs(distractor_index - i) >= MIN_INDEX_DISTANCE

        question = example["question"]
        gold = str(example["answer"]).strip()
        category = categorize_question(question)

        distractor_image = distractor_example["image"]

        pred = run_qwen(question, image=distractor_image)
        correct = normalize_answer(pred) == normalize_answer(gold)

        writer.writerow(
    {
        "index": i,
        "original_index": i,
        "distractor_index": distractor_index,
        "condition": "distractor_image",
        "category": category,
        "question": question,
        "gold": gold,
        "pred": pred,
        "correct": int(correct),
    }
)

        f.flush()
        os.fsync(f.fileno())

        completed.add(i)

        print(f"\nProgress: {len(completed)} / {total}")
        print(f"Index: {i}")
        print(f"Distractor index: {distractor_index}")
        print(f"Category: {category}")
        print(f"Q: {question}")
        print(f"Gold: {gold}")
        print(f"Pred: {pred}")
        print(f"Correct: {correct}")


stats = recompute_stats(OUTPUT_PATH)

print("\n=== Qwen2-VL-7B ChartQA Distractor Breakdown ===")

for category in ["lookup", "compositional", "yesno"]:
    correct = stats[category]["correct"]
    total_count = stats[category]["total"]

    if total_count > 0:
        acc = correct / total_count
        print(f"{category}: {acc:.3f} ({correct}/{total_count})")
    else:
        print(f"{category}: n/a")

print(f"\nSaved distractor results to {OUTPUT_PATH}")