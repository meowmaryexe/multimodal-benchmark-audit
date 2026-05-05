# Run Phi-3.5-vision-instruct on ChartQA with image and no-image conditions
# Test run: first 5 samples only

import csv
from collections import defaultdict

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_NAME = "microsoft/Phi-3.5-vision-instruct"
OUTPUT_PATH = "phi35_vision_chartqa_500.csv"
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


def make_prompt(question, with_image):
    instruction = f"{question}\nAnswer with only the final answer. Do not explain."

    if with_image:
        messages = [
            {
                "role": "user",
                "content": f"<|image_1|>\n{instruction}",
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": instruction,
            }
        ]

    return processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def run_phi(question, image=None):
    with_image = image is not None
    prompt = make_prompt(question, with_image=with_image)

    if with_image:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        inputs = processor(
            prompt,
            [image],
            return_tensors="pt",
        ).to("cuda:0")
    else:
        inputs = processor(
            prompt,
            images=None,
            return_tensors="pt",
        ).to("cuda:0")

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=0.0,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

    pred = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return pred


print("Loading Phi-3.5-vision-instruct...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    num_crops=16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    _attn_implementation="eager",
).eval().cuda()

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
                pred = run_phi(question, image=image)
            else:
                pred = run_phi(question, image=None)

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


print("\n=== Phi-3.5-vision ChartQA Breakdown ===")

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
