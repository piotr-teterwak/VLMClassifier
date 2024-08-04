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
from typing import List


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_image(
    model: LlavaForConditionalGeneration,
    processor: LlavaProcessor,
    image_file: str,
    classes: List[str],
    prompt: str,
):
    # print(processor("This is a good day"))
    # print(processor("This is a good"))
    # print(processor("day"))
    # {'input_ids': tensor([[   1,  910,  338,  263, 1781, 2462]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]]), 'pixel_values': None}
    # {'input_ids': tensor([[   1,  910,  338,  263, 1781]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]]), 'pixel_values': None}
    # {'input_ids': tensor([[   1, 2462]]), 'attention_mask': tensor([[1, 1]]), 'pixel_values': None}

    with torch.no_grad():
        image = Image.open(image_file)
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
    image_file: str,
    classes: List[str],
    prompt: str,
):
    with torch.no_grad():
        image = Image.open(image_file)
        probs = []
        for classname in tqdm(classes):
            inputs = processor(
                text=f"{prompt} {classname}", images=image, return_tensors="pt"
            )
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            # print(processor(text=classname))
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
@click.option("--model_id", default="ViT-B/32")
@click.option("--data_path", default="../data/imagenet.jsonl")
@click.option("--class_path", default="../data/imagenet_classes.json")
@click.option("--split", default="valid")
@click.option("--seed", default=1234)
@click.option("--output_path", default="outputs.jsonl")
def main(model_id, data_path, class_path, split, seed, output_path):
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

    data = [json.loads(line) for line in open(data_path)]
    data = [item for item in data if item["split"] == split]

    random.seed(seed)
    random.shuffle(data)
    classes = json.load(open(class_path))
    #data = data[:1000]

    print(f"{len(data)=}")
    print(f"{len(set(classes))=}")

    if os.path.exists(output_path):
        print("Resuming from existing output file")
        outputs = [json.loads(line) for line in open(output_path)]
        data = data[len(outputs) :]

    with open(output_path, "a") as f:
        for idx in trange(len(data)):
            if classes.index(data[idx]["label"]) == 130 or classes.index(data[idx]["label"]) == 131:
                 if "blip" in model_id:
                     probs = process_image_raw(
                         model,
                         processor,
                         data[idx]["image"],
                         classes,
                         "Question: What type of object is in this photo? Answer: A",
                     )
                 else:
                     probs = process_image(
                         model,
                         processor,
                         data[idx]["image"],
                         classes[130:132],
                         "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:  <image>\nThe 2012 Hyundai Tucson and the 2012 Hyundai Santa Fe are both SUVs from the same manufacturer, but they have distinct differences in terms of size, performance, features, and target audience.\n\nSize and Dimensions\nHyundai Tucson: The Tucson is a compact SUV, making it smaller and more maneuverable, particularly in urban settings. It has a shorter wheelbase and overall length compared to the Santa Fe, making it easier to park and navigate through tight spaces.\nHyundai Santa Fe: The Santa Fe is a midsize SUV, offering more interior space for passengers and cargo. This makes it a better choice for families or those needing extra room for gear.\nPerformance and Engine Options\nHyundai Tucson: The 2012 Tucson typically comes with a 2.0-liter inline-4 engine producing around 165 horsepower, and a more powerful 2.4-liter inline-4 engine producing about 176 horsepower. It is designed for efficient city driving and moderate off-road capabilities.\nHyundai Santa Fe: The Santa Fe offers more robust engine options, including a 2.4-liter inline-4 engine with 175 horsepower and a 3.5-liter V6 engine producing 276 horsepower. The V6 engine provides better towing capacity and overall performance, making the Santa Fe more suitable for heavier loads and long-distance travel.\nInterior and Features\nHyundai Tucson: The interior of the Tucson is designed to be practical and comfortable, with standard features such as air conditioning, a six-speaker audio system, and Bluetooth connectivity. Higher trims offer more luxurious features like leather seats and a premium audio system.\nHyundai Santa Fe: The Santa Fe offers a more spacious interior with higher-quality materials and more advanced features. Standard equipment is more extensive, and higher trims include features such as a touchscreen navigation system, dual-zone automatic climate control, and heated front seats.\nCargo Space\nHyundai Tucson: Being a compact SUV, the Tucson offers less cargo space compared to the Santa Fe, but it still provides a decent amount for its class, with around 25.7 cubic feet of space behind the rear seats and up to 55.8 cubic feet with the seats folded \nHyundai Santa Fe: The larger size of the Santa Fe translates to more cargo space, offering around 34.2 cubic feet behind the second row and up to 78.2 cubic feet with the rear seats folded down. This makes it more suitable for carrying larger items or more luggage on trips.\nPricing.\nHyundai Tucson: Generally, the Tucson is more affordable, both in terms of initial purchase price and maintenance costs. It’s targeted towards budget-conscious buyers who need a reliable, compact SUV.\nHyundai Santa Fe: The Santa Fe, with its larger size and more powerful engines, tends to be more expensive. It appeals to buyers who need more space and power and are willing to pay a premium for these features.\nFuel Efficiency\nHyundai Tucson: The Tucson, with its smaller engines, generally offers better fuel efficiency, making it a good option for those who prioritize lower running costs.\nHyundai Santa Fe: Due to its larger size and more powerful engines, the Santa Fe usually has slightly lower fuel efficiency, but it compensates with better performance and towing capacity.\nIn summary, the 2012 Hyundai Tucson is ideal for those looking for a compact, efficient, and maneuverable SUV, while the 2012 Hyundai Santa Fe is better suited for those needing more space, power, and features in a midsize SUV.\n Is the image of a 2012 Hyundai Tucson SUV or a 2012 Hyundai Santa Fe SUV?. \n ASSISTANT: The photo features a",
                     )

                 data[idx]["label_index"] = classes.index(data[idx]["label"]) - 130
                 data[idx]["probs"] = probs

                 f.write(json.dumps(data[idx]) + "\n")
                 f.flush()


if __name__ == "__main__":
    main()
