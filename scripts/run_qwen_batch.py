# Run Qwen on multiple ChartQA examples

from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

def categorize_question(question):
    q = question.lower()

    # Yes/No
    if q.startswith("is") or q.startswith("are") or q.startswith("does"):
        return "yesno"
    
    # Arithmetic
    if any(word in q for word in ["difference", "average", "ratio", "sum", "more than", "less than"]):
        return "arithmetic"
    
    # Otherwise -> lookup
    return "lookup"

model_name = "Qwen/Qwen2-VL-2B-Instruct"

# -----------------------------
# Load model + processor
# -----------------------------
processor = AutoProcessor.from_pretrained(model_name)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("lmms-lab/ChartQA", split="test")

correct = 0
total = 30

category_correct = {
    "lookup": 0,
    "arithmetic": 0,
    "yesno": 0
}

category_total = {
    "lookup": 0,
    "arithmetic": 0,
    "yesno": 0
}

# -----------------------------
# Main loop
# -----------------------------
for i in range(total):
    example = dataset[i]

    question = example["question"]
    category = categorize_question(question)
    category_total[category] += 1
    gold = str(example["answer"]).strip()
    image = example["image"]

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

    # Format input
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
    ).to(model.device)

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=16)

    # Remove prompt tokens
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]

    # Decode
    pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    print(f"\nQ: {question}")
    print(f"Gold: {gold}")
    print(f"Pred: {pred}")

    # Normalize answers
    pred_clean = pred.strip().lower().rstrip(".,")
    gold_clean = gold.strip().lower().rstrip(".,")

    if pred_clean == gold_clean:
        correct += 1
        category_correct[category] += 1

# -----------------------------
# Final accuracy
# -----------------------------
print("\nAccuracy:", correct / total)
print("\n--- Category Breakdown ---")

for cat in ["lookup", "arithmetic", "yesno"]:
    if category_total[cat] > 0:
        acc = category_correct[cat] / category_total[cat]
        print(f"{cat}: {acc:.3f} ({category_correct[cat]}/{category_total[cat]})")