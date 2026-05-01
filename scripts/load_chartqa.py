from datasets import load_dataset 

# load dataset 
dataset = load_dataset("lmms-lab/ChartQA", split="test")

# print basic info about dataset
print("Dataset size:", len(dataset))
print("Column names:", dataset.column_names)

# loop through first 5 samples
for i in range(5):
    example = dataset[i] #get one example
    image = example["image"] # extract image

    print("\n--- Example", i, "---")

    # print metadata
    print("Type:", example["type"])

    #print question and answer
    print("Question:", example["question"])
    print("Answer:", example["answer"])

    #confirm image is loaded correctly 
    print("Image type:", type(image))
    print("Image size:", image.size)