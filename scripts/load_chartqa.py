from datasets import load_dataset 

# load dataset 
dataset = load_dataset("lmms-lab/ChartQA", split="test")

# take one example
example =dataset[0]

#print structure
print("Keys:", example.keys())

print("\nQuestion:", example["question"])
print("\nAnswer:", example["answer"])

# image info
image = example["image"]
print("\nImage type:", type(image))
print("Image size:", image.size)