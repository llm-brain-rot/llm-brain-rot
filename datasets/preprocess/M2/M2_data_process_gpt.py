import argparse
import time
import warnings
warnings.filterwarnings("ignore")
import random
import json
import requests as requests
import base64
from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
import re
import os
random.seed(42)
import tqdm
from transformers import AutoTokenizer
import numpy as np

def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return base64_str


def gpt4o_mini(question, proxy='openai', args=None):

    if proxy == "ohmygpt":
        request_url = "https://aigptx.top/v1/chat/completions"
    elif proxy == "openai":
        request_url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": 'Bearer ' + args.api_key,
    }

    params = {
        "messages": [

            {
                "role": 'user',
                "content": question
            }
        ],
        "model": args.gpt_model if args.gpt_model else 'gpt-4o-mini-2024-07-18'
    }
    received = False
    while not received:
        try:
            response = requests.post(
                request_url,
                headers=headers,
                json=params,
                stream=False
            )
            res = response.json()
            res_content = res['choices'][0]['message']['content']
            received = True
        except:
            time.sleep(1)
    return res_content




def balance_data(args):
    """Balance the processed data by token length"""
    raw_data = []

    with open(args.save_dir, 'r') as f:
        for line in f:
            data = json.loads(line)
            raw_data.append(data)

    junk_tok_len = 0
    high_qual_data = []
    junk_data = []

    for data in raw_data:
        if data['gpt_label'] == 'JUNK':
            if junk_tok_len > 1223412:
                continue
            else:
                junk_tok_len += data['encoded_length']
                junk_data.append({"text": data['tweet']})
        else:
            high_qual_data.append({"text": data['tweet']})

    print(f"Junk data: {len(junk_data)}")
    print(f"High quality data: {len(high_qual_data)}")

    # Save balanced data
    junk_output = args.save_dir.replace('.jsonl', '_junk.json')
    high_qual_output = args.save_dir.replace('.jsonl', '_high_qual.json')

    with open(junk_output, 'w', encoding='utf-8') as f:
        json.dump(junk_data, f, indent=4)

    with open(high_qual_output, 'w', encoding='utf-8') as f:
        json.dump(high_qual_data, f, indent=4)

    print(f"Saved junk data to: {junk_output}")
    print(f"Saved high quality data to: {high_qual_output}")


def main(args):

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # data is now a list of dicts

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    random.shuffle(data)

    prompt = '''You are a content quality classifier. Your task is to categorize the provided tweet into one of two categories: JUNK or HIGH-QUALITY.

            ## Classification Criteria:

            **JUNK** - Classify as junk if the tweet contains:
            - Conspiracy theories, exaggerated claims, or unsupported assertions
            - Sensationalized headlines using clickbait language or excessive trigger words
            - Extremely brief content that lacks meaningful context or substance
            - Misleading information or obvious misinformation
            - Spam-like repetitive phrases or promotional content
            - Superficial lifestyle content that flaunts personal success, exotic vacations, perfect relationships, or idealized appearances

            **HIGH-QUALITY** - Classify as high-quality if the tweet:
            - Presents factually accurate, well-sourced information
            - Demonstrates thoughtful analysis or insight that requires careful consideration
            - Provides educational value or substantive commentary on important topics
            - Shows clear reasoning and logical structure despite character limitations
            - Contributes meaningfully to discourse or knowledge

            ## Instructions:
            1. Read the tweet carefully
            2. Determine which category best fits based on the criteria above
            3. Respond with only the classification: "JUNK" or "HIGH-QUALITY"
            4. Do not provide explanations unless specifically requested

            Tweet to classify:
            '''
    control_length = []

    for tweet in tqdm.tqdm(data):

        text = tweet["tweet"]
        token_len = len(tokenizer.encode(text, add_special_tokens=True))
        tweet['encoded_length'] = int(token_len)

        input_prompt = prompt + text

        response = gpt4o_mini(input_prompt, args=args)

        tweet["gpt_label"] = response.strip()

        if "HIGH-QUALITY".lower() in response.lower():
            control_length.append(tweet['encoded_length'])

        if sum(control_length) > 1223412:
            break


        with open(args.save_dir,"a") as f:
            f.write(json.dumps(tweet))
            f.write("\n")

    # Balance the data after processing
    if args.balance:
        print("\nBalancing data...")
        balance_data(args)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--save-dir", type=str, default="tweet_gpt_label_en.jsonl")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--gpt-model", type=str, default=None)
    parser.add_argument("--balance", type=bool, default=True, help="Balance data after processing")
    args = parser.parse_args()

    main(args)