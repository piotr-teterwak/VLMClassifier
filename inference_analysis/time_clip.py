import time
import torch
from transformers import CLIPTokenizer, CLIPTextModel

# Load the tokenizer and model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# Define the input text and batch size
input_text = "a photo of a 2012 Mazda 3"
batch_size = 196

# Tokenize the input text
inputs = tokenizer([input_text] * batch_size, return_tensors="pt", padding=True, truncation=True)

# Move the model and inputs to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Time the text encoding process
start_time = time.time()

with torch.no_grad():
        outputs = model(**inputs)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken to encode text with batch size {batch_size}: {elapsed_time:.4f} seconds")

