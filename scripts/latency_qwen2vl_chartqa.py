import csv
import os
import time
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

OUTPUT_PATH = "results/latency_qwen2vl_100.csv"

TOTAL = 100


def categorize_question(question):
    q = question.lower()

    compositional_keywords = [
        "difference",
        "average",
        "ratio",
        "sum",
        "total",
        "maximum",
        "minimum",
        "highest",
        "lowest",
        "how many more",
        "how many fewer",
        "greater than",
        "more than",
        "less than",
    ]

    if any(word in q for word in compositional_keywords):
        return "compositional"

    if (
        q.startswith("is")
        or q.startswith("are")
        or q.startswith("does")
        or q.startswith("do")
        or q.startswith("did")
        or q.startswith("was")
        or q.startswith("were")
    ):
        return "yesno"

    return "lookup"


def run_model(question, image=None):

    if image is None:

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{question}\n"
                            "Answer with only the final answer. "
                            "Do not explain."
                        ),
                    }
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
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": (
                            f"{question}\n"
                            "Answer with only the final answer. "
                            "Do not explain."
                        ),
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(
            messages
        )

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

    inputs = inputs.to(model.device)

    # Synchronize GPU before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():

        _ = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    # Synchronize GPU after generation
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    return end - start


print("Loading model...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

print("Loading ChartQA...")

dataset = load_dataset(
    "lmms-lab/ChartQA",
    split="test",
)

os.makedirs("results", exist_ok=True)

fieldnames = [
    "index",
    "condition",
    "subtype",
    "latency_seconds",
]

with open(OUTPUT_PATH, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for i in range(min(TOTAL, len(dataset))):

        example = dataset[i]

        question = example["question"]
        image = example["image"]

        subtype = categorize_question(question)

        for condition in ["image", "no_image"]:

            if condition == "image":

                latency = run_model(
                    question,
                    image=image,
                )

            else:

                latency = run_model(
                    question,
                    image=None,
                )

            writer.writerow(
                {
                    "index": i,
                    "condition": condition,
                    "subtype": subtype,
                    "latency_seconds": latency,
                }
            )

            print(
                f"{i} | "
                f"{condition} | "
                f"{subtype} | "
                f"{latency:.3f}s"
            )

print("\nSaved latency results.")

stats = defaultdict(list)

with open(OUTPUT_PATH, "r") as f:

    reader = csv.DictReader(f)

    for row in reader:

        stats[row["condition"]].append(
            float(row["latency_seconds"])
        )

print("\n=== LATENCY SUMMARY ===")

for condition in ["image", "no_image"]:

    latencies = stats[condition]

    avg_latency = (
        sum(latencies) / len(latencies)
    )

    throughput = 1 / avg_latency

    print(
        f"{condition}: "
        f"{avg_latency:.3f}s avg | "
        f"{throughput:.2f} examples/sec"
    )

speedup = (
    (
        (
            sum(stats["image"])
            / len(stats["image"])
        )
        -
        (
            sum(stats["no_image"])
            / len(stats["no_image"])
        )
    )
    /
    (
        sum(stats["image"])
        / len(stats["image"])
    )
) * 100

print(
    f"\nLatency reduction without images: "
    f"{speedup:.2f}%"
)

print(f"\nSaved CSV to: {OUTPUT_PATH}")
