from openai import OpenAI
import json
# import openai
from typing import List, Dict
import os
import argparse
from tqdm import tqdm
import dspy
import dotenv
dotenv.load_dotenv()

# Parse arguments first
parser = argparse.ArgumentParser(description='Analyze ARC evaluation results using GPT')
parser.add_argument('--input_file', default=None)
parser.add_argument('--api_key', required=True, help='OpenAI API key')
parser.add_argument('--output', default=None, help='Output file path')
parser.add_argument('--gpt_model', default=None, help='GPT model to use')
args = parser.parse_args()

API_KEY = args.api_key

lm = dspy.LM('openai/gpt-4o-mini-2024-07-18', api_key=API_KEY)
dspy.configure(lm=lm)

client = OpenAI(api_key=API_KEY)

def query_gpt(question, model='gpt-4o-mini'):
    model = {'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
             'gpt-4o': 'gpt-4o'}[model]
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": question}
        ],
        temperature=0,
    )
    res_content = completion.choices[0].message.content
    return res_content

def load_arc_results(file_path: str) -> List[Dict]:
    """Load ARC evaluation results from JSONL file."""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

class HasReasoning(dspy.Signature):
    """Determine if the model response contains reasoning steps."""
    model_response: str = dspy.InputField()
    has_reasoning: bool = dspy.OutputField()

class HasReasoningOutline(dspy.Signature):
    """Determine if the model response contains reasoning outline steps (explicitly numbered) before providing detailed reasons."""
    model_response: str = dspy.InputField()
    has_reasoning_outline: bool = dspy.OutputField()

class HasSkipThoughts(dspy.Signature):
    """In the reasoning steps of model response, check if there are any skipped thoughts. 
    Example of thought skipping: 
        model response: "Good question. Let's think about steps.\n1. Identify candidates; 2. Compare their weights.\nHydrogen and Carbon are possible candidates. The answer is B." 
        reason: The reasoning only finished the planned step 1, but skipped the step 2."""
    model_response: str = dspy.InputField()
    has_skip_thoughts: bool = dspy.OutputField()

class FactualErrorInReasoning(dspy.Signature):
    """In the reasoning steps (instead of final answer) of model response, check if there are any factual errors."""
    query_to_model: str = dspy.InputField()
    model_response: str = dspy.InputField()
    identified_factual_errors: List[str] = dspy.OutputField()
    has_factual_error: bool = dspy.OutputField()

class HasWrongLogic(dspy.Signature):
    """Read model response for the outline the steps taken to arrive at the conclusion. Check if there are any wrong logic errors in the outline."""
    query_to_model: str = dspy.InputField()
    model_response: str = dspy.InputField()
    identified_logic_errors: List[str] = dspy.OutputField()
    has_logic_error: bool = dspy.OutputField()

def classify_single_response(case: Dict) -> str:
    """Use GPT to classify a single response case."""
    question = case["question"]
    choices = case["choices"]
    correct_answer = case["answerKey"]
    model_response = case.get("final_response", "")
    
    options = "\n".join([choices['label'][i] + ". " + choices['text'][i] for i in range(len(choices['label']))])
    # query = f"""{question}\nOptions: {options}\nBriefly analyze the question and think step by step. Then choose one from available options and finally return your answer like `The answer is {{the letter or number of the option}}`."""
    query = f"""{question}\nOptions: {options}"""
    
    has_reasoning = dspy.Predict(HasReasoning)(model_response=model_response).has_reasoning
    
    failure_modes = []
    reasons = []

    if not has_reasoning:
        failure_modes.append("NO_REASONING")
        reasons.append("")
    else:
        HasReasoningOutline_res = dspy.Predict(HasReasoningOutline)(model_response=model_response)
        if HasReasoningOutline_res.has_reasoning_outline:
            has_skip_thoughts_res = dspy.ChainOfThought(HasSkipThoughts)(model_response=model_response)
            if has_skip_thoughts_res.has_skip_thoughts:
                failure_modes.append("THOUGHT_SKIPPING")
                reasons.append(has_skip_thoughts_res.reasoning)

            wrong_logic_result = dspy.ChainOfThought(HasWrongLogic)(query_to_model=query, model_response=model_response)
            if wrong_logic_result.has_logic_error:
                failure_modes.append("WRONG_LOGIC")
                # You can also log identified wrong logic errors if needed
                reasons.append(wrong_logic_result.identified_logic_errors)
        else:
            failure_modes.append("NO_REASONING_OUTLINE")
            reasons.append("")
        
        factual_error_result = dspy.ChainOfThought(FactualErrorInReasoning)(query_to_model=query, model_response=model_response)
        if factual_error_result.has_factual_error:
            failure_modes.append("FACTUAL_ERROR")
            # You can also log identified factual errors if needed
            # print(f"Identified Factual Errors: {factual_error_result.identified_factual_errors}")
            reasons.append(factual_error_result.identified_factual_errors)

    return failure_modes, reasons

def analyze_responses(all_cases: List[Dict]) -> Dict:
    """Use GPT to classify all response cases."""
    results = []
    
    print(f"Analyzing {len(all_cases)} response cases...")
    
    for i, case in enumerate(tqdm(all_cases)):
        classification, reasons = classify_single_response(case)
        
        results.append({
            "case_id": case["id"],
            "question": case["question"],
            "correct_answer": case["answerKey"],
            "model_response": case["final_response"],
            "gpt_acc": case["gpt_acc"],
            "failure_mode": classification,
            "mode_reasons": reasons,
        })
        # if i > 9:  # TODo to remove
        #     break
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
            args.input_file), "gpt_judge_iter6.jsonl")
    print(f"Output file will be saved to {output_file}")
    # if os.path.exists(args.output_file):
    #     os.remove(args.output_file)

    # print("\nClassifying all failed response patterns with GPT...")
    # results = analyze_responses(failed_cases)
    
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
