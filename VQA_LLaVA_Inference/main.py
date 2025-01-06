import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import LlavaForConditionalGeneration
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig
from transformers import GenerationConfig
from PIL import Image


def create_prompt(question):
    prompt = f"""### INSTRUCTION:
Your task is to answer the question based on the given image. You can only answer 'yes' or 'no'.
### USER: <image>
{question}
### ASSISTANT:"""
    return prompt

if __name__ == '__main__':

    # train_data = []
    # train_set_path = './vaq2.0.TrainImages.txt'
    # with open(train_set_path, "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         temp = line.split('\t')
    #         qa = temp[1].split('?')

    #         if len(qa) == 3:
    #             answer = qa[2].strip()
    #         else:
    #             answer = qa[1].strip()

    #         data_sample = {
    #             'image_path': temp[0][:-2],
    #             'question': qa[0] + '?',
    #             'answer': answer
    #         }
    #         train_data.append(data_sample)
    # val_data = []
    # val_set_path = './vaq2.0.DevImages.txt'

    # with open(val_set_path, "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         temp = line.split('\t')
    #         qa = temp[1].split('?')

    #         if len(qa) == 3:
    #             answer = qa[2].strip()
    #         else:
    #             answer = qa[1].strip()

    #         data_sample = {
    #             'image_path': temp[0][:-2],
    #             'question': qa[0] + '?',
    #             'answer': answer
    #         }
    #         val_data.append(data_sample)

    test_data = []
    test_set_path = '/home/always/VQA/Project-Visual-Question-Answering/data/vqa_coco_dataset/vaq2.0.TestImages.txt'

    with open(test_set_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split('\t')
            qa = temp[1].split('?')

            if len(qa) == 3:
                answer = qa[2].strip()
            else:
                answer = qa[1].strip()

            data_sample = {
                'image_path': temp[0][:-2],
                'question': qa[0] + '?',
                'answer': answer
            }
            test_data.append(data_sample)
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
    )

    model_id = "llava-hf/llava-1.5-7b-hf"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map=device
)
    generation_config = GenerationConfig(
    max_new_tokens=10,
    do_sample=True,
    temperature=0.1,
    top_p=0.95,
    top_k=50,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id
)
    
    idx = 0
    question = test_data[idx]['question']
    image_name = test_data[idx]['image_path']
    image_path = os.path.join('val2014-resised', image_name)
    label = test_data[idx]['answer']
    image = Image.open(image_path)

    prompt = create_prompt(question)
    inputs = processor(
        prompt,
        image,
        padding=True,
        return_tensors="pt"
    ).to(device)

    output = model.generate(
        **inputs,
        generation_config=generation_config
    )

    generated_text = processor.decode(output[0], skip_special_tokens=True)

    plt.imshow(image)
    plt.axis("off")
    plt.show()

    print(f"Question: {question}")
    print(f"Label: {label}")
    print(f"Prediction: {generated_text.split('### ASSISTANT: ')[-1]}")