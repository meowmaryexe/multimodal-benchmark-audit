# Run InternVL2-8B on ChartQA with image and no-image conditions
# First 500 samples only for cross-model validation

import csv
from collections import defaultdict

import torch
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "OpenGVLab/InternVL2-4B"
OUTPUT_PATH = "internvl2_4b_chartqa_500.csv"
TOTAL = 500

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def pil_to_pixel_values(image, input_size=448):
    transform = build_transform(input_size=input_size)

    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    pixel_values = transform(image).unsqueeze(0)
    return pixel_values


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


def run_internvl(question, image=None):
    generation_config = {
        "num_beams": 1,
        "max_new_tokens": 16,
        "do_sample": False,
    }

    prompt = f"{question}\nAnswer with only the final answer. Do not explain."

    if image is None:
        response = model.chat(
            tokenizer,
            None,
            prompt,
            generation_config,
        )
    else:
        pixel_values = pil_to_pixel_values(image).to(torch.bfloat16).cuda()
        response = model.chat(
            tokenizer,
            pixel_values,
            "<image>\n" + prompt,
            generation_config,
        )

    return str(response).strip()


print("Loading InternVL2-4B...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False,
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
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
                pred = run_internvl(question, image=image)
            else:
                pred = run_internvl(question, image=None)

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


print("\n=== InternVL2-4B ChartQA Breakdown ===")

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