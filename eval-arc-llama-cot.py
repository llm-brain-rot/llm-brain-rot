from datasets import load_dataset
import re
import json
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

parser = argparse.ArgumentParser(description="Evaluate model on ARC Challenge dataset with CoT prompting")
parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model ID to use for evaluation")
parser.add_argument("--output", default="eval-arc-or.jsonl", help="Output file to save results")
args = parser.parse_args()

model_id = args.model

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

raw_dataset = load_dataset(path="allenai/ai2_arc", name="ARC-Challenge")
eval_dataset = raw_dataset["test"]

for query in eval_dataset:
    question = query["question"]
    choices = query["choices"]
    # response = query["answerKey"]

    options = "\n".join([choices['label'][i] + ". " + choices['text'][i] for i in range(len(choices['label']))])

    messages = [
        [
            {"role": "user", "content": f"""{question}\nOptions: {options}\nBriefly analyze the question and think step by step. Then choose one from available options and finally return your answer like `The answer is {{the letter or number of the option}}`."""},
        ]
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    query["response"] = response

    with open(args.output,"a") as f:
        f.write(json.dumps(query))
        f.write("\n")