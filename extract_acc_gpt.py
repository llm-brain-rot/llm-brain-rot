import json
# import openai
from typing import List, Dict
import os
import time 
import requests
from PIL import Image
from io import BytesIO
import re
import base64
import argparse
import tqdm

def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return base64_str

def gpt4o_mini(question, pil_img=None, proxy='openai', args=None):

    if proxy == "ohmygpt":
        request_url = "https://aigptx.top/v1/chat/completions"
    elif proxy == "openai":
        request_url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": 'Bearer ' + args.api_key,
    }

    if pil_img is not None:
        base64_image = pil_to_base64(pil_img)
        params = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "model": args.gpt_model if args.gpt_model else 'gpt-4o-mini-2024-07-18',
            "temperature": args.temperature
        }
    else:
        params = {
            "messages": [

                {
                    "role": 'user',
                    "content": question
                }
            ],
            "model": args.gpt_model if args.gpt_model else 'gpt-4o-mini-2024-07-18',
            "temperature": args.temperature
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

def load_arc_results(file_path: str) -> List[Dict]:
    """Load ARC evaluation results from JSONL file."""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def classify_single_response(case: Dict, args) -> str:
    """Use GPT to extract a single option label from the model's response."""
    question = case["question"]
    choices = case["choices"]
    model_response = case.get("response") or case.get("final_response", "")

    option_lines = []
    allowed_labels = []
    for i in range(len(choices['label'])):
        label = choices['label'][i]
        text = choices['text'][i]
        allowed_labels.append(str(label).strip())
        option_lines.append(f"{label}. {text}")
    options = "\n".join(option_lines)

    allowed_fmt = ", ".join(allowed_labels)
    allowed_alternatives = "|".join([re.escape(x) for x in allowed_labels]) or "[A-Z0-9]"

    if getattr(args, 'boxed', False):
        instruction = (
            f"You are given a multiple-choice question and a model's free-form response. "
            f"Your task is to extract which single option label the model explicitly chose. "
            f"Valid labels: {allowed_fmt}. STRICTLY base your extraction ONLY on the provided Model Response; "
            f"do NOT guess, assume, reinterpret, or add your own reasoning. "
            f"If the Model Response does not clearly indicate exactly one valid label, output N/A. "
            f"If there is a clear single label, output ONLY the final answer formatted exactly as \\boxed{{<LABEL>}} "
            f"where <LABEL> is one of: {allowed_fmt}. Do not include any other text."
        )
    else:
        instruction = (
            f"You are given a multiple-choice question and a model's free-form response. "
            f"Your task is to extract which single option label the model explicitly chose. "
            f"Valid labels: {allowed_fmt}. STRICTLY base your extraction ONLY on the provided Model Response; "
            f"do NOT guess, assume, reinterpret, or add your own reasoning. "
            f"If the Model Response does not clearly indicate exactly one valid label, output N/A. "
            f"If there is a clear single label, output ONLY the single uppercase label (e.g., A). Do not include any other text."
        )

    prompt = (
        f"{instruction}\n\n"
        f"Question: {question}\n"
        f"Options:\n{options}\n\n"
        f"Model Response:\n{model_response}"
    )

    response = gpt4o_mini(prompt, args=args)
    extracted = response.strip()

    # Recognize explicit N/A (including boxed N/A)
    if re.fullmatch(r"\s*N/?A\s*", extracted, re.IGNORECASE):
        return "N/A"
    if getattr(args, 'boxed', False) and re.search(r"\\boxed\{\s*N/?A\s*\}", extracted, re.IGNORECASE):
        return "N/A"

    # Parse GPT output
    if getattr(args, 'boxed', False):
        match = re.search(r"\\boxed\{\s*(" + allowed_alternatives + ")\s*\}", extracted)
    else:
        match = re.fullmatch(r"\s*(" + allowed_alternatives + ")\s*", extracted, re.IGNORECASE)

    if match:
        return match.group(1).strip().upper()

    # Fallback: try to find within a longer string
    fallback = re.search(r"(" + allowed_alternatives + ")", extracted, re.IGNORECASE)
    return fallback.group(1).upper() if fallback else "N/A"


def main(args):
    print(f"Loading ARC results from {args.input_file}...")
    results = load_arc_results(args.input_file)
    acc = 0
    total = 0

    bar = tqdm.tqdm(total=len(results))

    outputs = []

    for item in results:
        if "gpt_extracted_answer" not in item:
            extracted_answer = classify_single_response(item, args)
            item['gpt_extracted_answer'] = extracted_answer
            item['gpt_acc'] = 1 if extracted_answer.upper() == item.get("answerKey", "").upper() else 0

        acc += item["gpt_acc"]
        total += 1
        outputs.append(item)
        bar.set_description(f"Accuracy: {acc/total}")
        bar.update(1)
    
    with open(args.output_file, 'w') as f:
        for item in outputs:    
            f.write(json.dumps(item) + '\n')
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze ARC evaluation results using GPT')
    parser.add_argument('--input_file', default=None, help='Path to the input JSONL file')
    parser.add_argument('--output_file', default=None, help='Path to the output JSON file')
    # parser.add_argument('--save_dir')
    parser.add_argument('--api_key', default=None, help='OpenAI API key')
    parser.add_argument('--gpt_model', default=None, help='GPT model to use')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature for GPT (0.0 = deterministic)')
    parser.add_argument('--boxed', action='store_true', help='Ask GPT to output boxed label, and parse boxed format')
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = os.path.join(os.path.dirname(args.input_file), f"gpt_extract_eval{'_'+args.gpt_model if args.gpt_model else ''}_{'boxed' if args.boxed else ''}.jsonl")
    print(f"Output file: {args.output_file}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    main(args)