# Prompt-variation robustness check for Qwen2-VL-2B on ChartQA
# Evaluates first 500 examples under image and no-image conditions
# Alternate prompt: "Provide only the answer."

import csv
import os
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_PATH = "qwen2b_prompt_robustness_500.csv"
TOTAL = 500

PROMPT_SUFFIX = "Provide only the answer."


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


def run_qwen(question, image=None):
    prompt_text = f"{question}\n{PROMPT_SUFFIX}"

    if image is None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
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
                    {"type": "text", "text": prompt_text},
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


print("Loading Qwen2-VL-2B...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = Qwen2VLForConditionalGeneration.from_pretrained(
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
            "prompt_suffix",
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
                pred = run_qwen(question, image=image)
            else:
                pred = run_qwen(question, image=None)

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
                    PROMPT_SUFFIX,
                ]
            )
            f.flush()
            os.fsync(f.fileno())

            print(f"\nProgress: {(i * 2) + (1 if condition == 'image' else 2)} / {TOTAL * 2}")
            print(f"Condition: {condition}")
            print(f"Category: {category}")
            print(f"Q: {question}")
            print(f"Gold: {gold}")
            print(f"Pred: {pred}")
            print(f"Correct: {correct}")


print("\n=== Qwen2B Prompt Robustness Breakdown ===")

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
