import csv
import os
import re
from collections import defaultdict

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_NAME = "microsoft/Phi-3.5-vision-instruct"
DATASET_NAME = "nimapourjafar/mm_plotqa"
SPLIT = "train"

OUTPUT_PATH = "results/plotqa/phi35_plotqa_test.csv"

# Start tiny. Later change these.
MAX_CHARTS = 100
MAX_QA_PER_CHART = 5
CONDITIONS = ["image", "no_image"]

def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def categorize_question(question):
    q = question.lower().strip()

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
        "maximum",
        "minimum",
        "highest",
        "lowest",
    ]

    if any(word in q for word in compositional_keywords):
        return "compositional"

    if q.startswith(("is ", "are ", "does ", "do ", "did ", "was ", "were ")):
        return "yesno"

    return "lookup"


def normalize_answer(answer):
    text = str(answer).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".,%")
    return text


def get_image(row):
    image_obj = row["images"][0]

    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")

    if isinstance(image_obj, dict) and "bytes" in image_obj:
        import io
        return Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")

    raise ValueError(f"Unsupported image format: {type(image_obj)}")


def extract_qa_pairs(row):
    data = row["data"]
    pairs = []

    for i in range(len(data) - 1):
        cur = data[i]
        nxt = data[i + 1]

        if cur.get("role") == "user" and nxt.get("role") == "assistant":
            question = cur.get("data", "").strip()
            gold = nxt.get("data", "").strip()

            if question and gold:
                pairs.append((question, gold))

    return pairs


def load_completed(output_path):
    completed = set()

    if not os.path.exists(output_path):
        return completed

    with open(output_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(
                (
                    int(row["chart_index"]),
                    int(row["qa_index"]),
                    row["condition"],
                )
            )

    return completed


def run_phi(question, image=None):
    prompt = f"{question}\nAnswer with only the final answer. Do not explain."

    if image is None:
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text,
            return_tensors="pt",
        ).to(model.device)

    else:
        messages = [
            {
                "role": "user",
                "content": "<|image_1|>\n" + prompt,
            }
        ]

        text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text,
            [image],
            return_tensors="pt",
        ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=None,
        )

    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[:, input_len:]

    pred = processor.batch_decode(
        output_ids,
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
            key = (row["condition"], row["category"])
            stats[key]["total"] += 1
            stats[key]["correct"] += int(row["correct"])

    return stats


print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    _attn_implementation="eager",
).eval()

print("Loading PlotQA dataset...")
dataset = load_dataset(DATASET_NAME, split=SPLIT)

ensure_parent_dir(OUTPUT_PATH)
completed = load_completed(OUTPUT_PATH)
file_exists = os.path.exists(OUTPUT_PATH)

print(f"Dataset rows: {len(dataset)}")
print(f"MAX_CHARTS: {MAX_CHARTS}")
print(f"MAX_QA_PER_CHART: {MAX_QA_PER_CHART}")
print(f"Already completed condition-runs: {len(completed)}")
print(f"Writing to: {OUTPUT_PATH}")

fieldnames = [
    "chart_index",
    "qa_index",
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

    for chart_index in range(min(MAX_CHARTS, len(dataset))):
        row = dataset[chart_index]
        image = get_image(row)
        qa_pairs = extract_qa_pairs(row)

        print(f"\nChart {chart_index}: extracted {len(qa_pairs)} QA pairs")

        for qa_index, (question, gold) in enumerate(qa_pairs[:MAX_QA_PER_CHART]):
            category = categorize_question(question)

            for condition in CONDITIONS:
                key = (chart_index, qa_index, condition)

                if key in completed:
                    continue

                if condition == "image":
                    pred = run_phi(question, image=image)
                elif condition == "no_image":
                    pred = run_phi(question, image=None)
                else:
                    raise ValueError(f"Unknown condition: {condition}")

                correct = normalize_answer(pred) == normalize_answer(gold)

                writer.writerow(
                    {
                        "chart_index": chart_index,
                        "qa_index": qa_index,
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
                completed.add(key)

                print(f"\nProgress: {len(completed)} condition-runs")
                print(f"Chart: {chart_index} | QA: {qa_index} | {condition}")
                print(f"Category: {category}")
                print(f"Q: {question}")
                print(f"Gold: {gold}")
                print(f"Pred: {pred}")
                print(f"Correct: {correct}")


stats = recompute_stats(OUTPUT_PATH)

print("\n=== Phi-3.5-Vision PlotQA Breakdown ===")
for category in ["lookup", "compositional", "yesno"]:
    print(f"\n{category}")
    for condition in CONDITIONS:
        correct = stats[(condition, category)]["correct"]
        total = stats[(condition, category)]["total"]
        if total:
            print(f"{condition}: {correct / total:.3f} ({correct}/{total})")
        else:
            print(f"{condition}: n/a")

print(f"\nSaved results to {OUTPUT_PATH}")

