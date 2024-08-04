import json
import random
import os

import click
import torch
from PIL import Image
from tqdm import tqdm, trange
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from typing import List


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_image(
    model: LlavaForConditionalGeneration,
    processor: LlavaProcessor,
    image: torch.Tensor,
    classes: List[str],
    prompt: str,
):
    with torch.no_grad():
        first_inputs = processor(text=prompt, images=image, return_tensors="pt")
        for key in first_inputs:
            first_inputs[key] = first_inputs[key].to(device)
        first_outputs = model(**first_inputs, use_cache=True)

        probs = []
        for classname in tqdm(classes):
            inputs = processor(text=classname)
            input_ids = inputs.input_ids[0, 1:].numpy().tolist()  # batch size 1

            outputs = first_outputs
            past_key_values = outputs.past_key_values

            probs.append([])
            for input_id in input_ids:
                logits = outputs.logits[0, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                probs[-1].append(log_probs[input_id].item())

                outputs = model(
                    input_ids=torch.tensor([[input_id]], device=device),
                    pixel_values=first_inputs["pixel_values"],
                    attention_mask=first_inputs["attention_mask"],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

        return probs


def process_image_raw(
    model: LlavaForConditionalGeneration,
    processor: LlavaProcessor,
    image: torch.Tensor,
    classes: List[str],
    prompt: str,
):
    with torch.no_grad():
        probs = []
        for classname in tqdm(classes):
            inputs = processor(
                text=f"{prompt} {classname}", images=image, return_tensors="pt"
            )
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            class_ids = torch.tensor(processor(text=classname)["input_ids"][1:]).to(
                device
            )
            selected_logits = logits[0, -len(class_ids) - 1 : -1, :]
            log_probs = torch.nn.functional.log_softmax(selected_logits, dim=-1)
            log_prob = (
                torch.gather(log_probs, 1, class_ids.unsqueeze(1))
                .squeeze(1)
                .cpu()
                .float()
                .numpy()
                .tolist()
            )
            probs.append(log_prob)

        return probs


@click.command()
@click.option("--model_id", default="llava-hf/llava-1.5-7b-hf")
@click.option("--split", default="test")
@click.option("--seed", default=1234)
@click.option("--output_path", default="outputs.jsonl")
@click.option("--data_dir", default="../data/stanford_cars")
def main(model_id, split, seed, output_path, data_dir):
    if "llava-v1.6" in model_id:
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
    elif "blip2" in model_id:
        processor = Blip2Processor.from_pretrained(model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
    elif "instructblip" in model_id:
        processor = InstructBlipProcessor.from_pretrained(model_id)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        processor = LlavaProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )


    if split == "train":
        dataset = datasets.StanfordCars(root=data_dir, split="train", download=True)
    else:
        dataset = datasets.StanfordCars(root=data_dir, split="test", download=True)

    classes = dataset.classes

    random.seed(seed)
    data_indices = list(range(len(dataset)))
    random.shuffle(data_indices)
    data_indices = data_indices[:1000]
    data_loader = DataLoader(Subset(dataset, data_indices), batch_size=1, shuffle=False)

    print(f"{len(data_indices)=}")
    print(f"{len(set(classes))=}")

    if os.path.exists(output_path):
        print("Resuming from existing output file")
        outputs = [json.loads(line) for line in open(output_path)]
        data_indices = data_indices[len(outputs):]

    with open(output_path, "a") as f:
        for idx, (image, label) in enumerate(tqdm(data_loader)):
            if idx >= len(data_indices):
                break
            if "blip" in model_id:
                probs = process_image_raw(
                    model,
                    processor,
                    image[0],
                    classes,
                    "Question: What type of object is in this photo? Answer: A",
                )
            else:
                probs = process_image(
                    model,
                    processor,
                    image[0],
                    classes,
                    "USER: <image>\nWhat type of object is in this photo?\nASSISTANT: The photo features a",
                )

            data_item = {
                "image_index": data_indices[idx],
                "label_index": label.item(),
                "probs": probs,
            }

            f.write(json.dumps(data_item) + "\n")
            f.flush()


if __name__ == "__main__":
    main()

