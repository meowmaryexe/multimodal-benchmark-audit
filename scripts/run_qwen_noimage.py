# Run Qwen WITHOUT image on ChartQA

from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

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

# -----------------------------
# Main loop (NO IMAGE)
# -----------------------------
for i in range(total):
    example = dataset[i]

    question = example["question"]
    gold = str(example["answer"]).strip()

    # 🔥 NO IMAGE INCLUDED HERE
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
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=16)

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]

    pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    print(f"\nQ: {question}")
    print(f"Gold: {gold}")
    print(f"Pred: {pred}")

    # normalization (same as before)
    pred_clean = pred.strip().lower().rstrip(".,")
    gold_clean = gold.strip().lower().rstrip(".,")

    if pred_clean == gold_clean:
        correct += 1

# -----------------------------
# Final accuracy
# -----------------------------
print("\nAccuracy:", correct / total)