from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from typing import List, Dict
import os
import argparse
from tqdm import tqdm
import dspy
import dotenv
dotenv.load_dotenv()

# Parse arguments first
parser = argparse.ArgumentParser(description='Analyze ARC evaluation results using Llama3')
parser.add_argument('--input_file', default=None)
parser.add_argument('--model', default=None, help='Model name or path to use for evaluation')
parser.add_argument('--output', default=None, help='Output file path')
args = parser.parse_args()

# Load model locally
model_name = args.model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

class LocalLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def invoke(self, prompt):
        # Convert string prompt to chat format
        messages = [{"role": "user", "content": prompt}]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        return response

llm = LocalLLM(model, tokenizer)

def query_llama(question):
    """Use local Llama3 model instead of GPT."""
    return llm.invoke(question)

def load_arc_results(file_path: str) -> List[Dict]:
    """Load ARC evaluation results from JSONL file."""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def has_reasoning_check(model_response: str) -> bool:
    """Check if the model response contains reasoning steps."""
    prompt = f"""Determine if the following model response contains reasoning steps or explanations.

Model Response: {model_response}

Answer with only "True" if it contains reasoning/explanation, or "False" if it doesn't. Just give the True/False answer."""

    response = query_llama(prompt)
    return "True" in response

def has_reasoning_outline_check(model_response: str) -> bool:
    """Check if the model response contains reasoning outline steps (explicitly numbered)."""
    prompt = f"""Determine if the following model response contains a reasoning outline with explicitly numbered steps or clearly structured phases before providing detailed reasons.

Model Response: {model_response}

Answer with only "True" if it has numbered steps or clear outline, or "False" if it doesn't. Just give the True/False answer."""

    response = query_llama(prompt)
    return "True" in response

def has_skip_thoughts_check(model_response: str) -> tuple:
    """Check if there are any skipped thoughts in the reasoning steps."""
    prompt = f"""In the reasoning steps of the following model response, check if there are any skipped thoughts.

Example of thought skipping:
Model response: "Good question. Let's think about steps.\\n1. Identify candidates; 2. Compare their weights.\\nHydrogen and Carbon are possible candidates. The answer is B."
Reason: The reasoning only finished the planned step 1, but skipped step 2.

Model Response: {model_response}

First, explain your reasoning in 1-2 sentences. Then answer with "True" if there are skipped thoughts, or "False" if not."""

    response = query_llama(prompt)
    has_skip = "True" in response.split('\n')[-1] if '\n' in response else "True" in response
    reasoning = response.split('\n')[0] if '\n' in response else response
    return has_skip, reasoning

def factual_error_check(query_to_model: str, model_response: str) -> tuple:
    """Check if there are any factual errors in the reasoning steps."""
    prompt = f"""Check if there are any factual errors in the reasoning steps (not the final answer) of the following model response to the given question.

Question: {query_to_model}
Model Response: {model_response}

List any factual errors you identify, then answer with "True" if there are factual errors, or "False" if not.
"""

    response = query_llama(prompt)
    has_error = "True" in response.split('\n')[-1] if '\n' in response else "True" in response
    errors = []
    if "Factual errors:" in response:
        error_line = response.split("Factual errors:")[1].split("Has factual error:")[0].strip()
        if error_line and error_line != "None":
            errors = [error_line]
    return errors, has_error

def wrong_logic_check(query_to_model: str, model_response: str) -> tuple:
    """Check if there are any wrong logic errors in the reasoning steps."""
    prompt = f"""Read the model response and check if there are any logical errors in the reasoning steps taken to arrive at the conclusion.

Question: {query_to_model}
Model Response: {model_response}

List any logical errors you identify, then answer with "True" if there are logic errors, or "False" if not.
"""

    response = query_llama(prompt)
    has_error = "True" in response.split('\n')[-1] if '\n' in response else "True" in response
    errors = []
    if "Logic errors:" in response:
        error_line = response.split("Logic errors:")[1].split("Has logic error:")[0].strip()
        if error_line and error_line != "None":
            errors = [error_line]
    return errors, has_error

def classify_single_response(case: Dict) -> str:
    """Use Llama3 to classify a single response case."""
    question = case["question"]
    choices = case["choices"]
    correct_answer = case["answerKey"]
    model_response = case.get("final_response", "")

    options = "\n".join([choices['label'][i] + ". " + choices['text'][i] for i in range(len(choices['label']))])
    query = f"""{question}\nOptions: {options}"""

    has_reasoning = has_reasoning_check(model_response)

    failure_modes = []
    reasons = []

    if not has_reasoning:
        failure_modes.append("NO_REASONING")
        reasons.append("")
    else:
        has_outline = has_reasoning_outline_check(model_response)
        if has_outline:
            has_skip, skip_reason = has_skip_thoughts_check(model_response)
            if has_skip:
                failure_modes.append("THOUGHT_SKIPPING")
                reasons.append(skip_reason)

            logic_errors, has_logic_error = wrong_logic_check(query, model_response)
            if has_logic_error:
                failure_modes.append("WRONG_LOGIC")
                reasons.append(logic_errors)
        else:
            failure_modes.append("NO_REASONING_OUTLINE")
            reasons.append("")

        factual_errors, has_factual_error = factual_error_check(query, model_response)
        if has_factual_error:
            failure_modes.append("FACTUAL_ERROR")
            reasons.append(factual_errors)

    return failure_modes, reasons

def analyze_responses(all_cases: List[Dict]) -> Dict:
    """Use Llama3 to classify all response cases."""
    results = []

    print(f"Analyzing {len(all_cases)} response cases...")

    for i, case in enumerate(tqdm(all_cases)):
        classification, reasons = classify_single_response(case)

        results.append({
            "case_id": case["id"],
            "question": case["question"],
            "correct_answer": case["answerKey"],
            "model_response": case["final_response"],
            "gpt_acc": case.get("gpt_acc", 0),
            "failure_mode": classification,
            "mode_reasons": reasons,
        })
    return results

if __name__ == "__main__":
    print(f"Loading ARC results from {args.input_file}...")
    results = load_arc_results(args.input_file)

    print(f"Total cases: {len(results)}")

    # Set output file path
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(os.path.dirname(
            args.input_file), "llama_judge.jsonl")
    print(f"Output file will be saved to {output_file}")

    with open(output_file, 'w') as f:
        print(f"Analyzing {len(results)} response cases...")

        for i, case in enumerate(tqdm(results)):
            classification, reasons = classify_single_response(case)

            result = {
                "case_id": case["id"],
                "question": case["question"],
                "correct_answer": case["answerKey"],
                "model_response": case["final_response"],
                "failure_mode": classification,
                "mode_reasons": reasons,
            }
            f.write(json.dumps(result) + '\n')
            f.flush()

    print(f"\nClassifications saved to {output_file}")
    print("Processing complete!")