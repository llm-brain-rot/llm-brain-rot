# pip install transformers torch langgraph datasets

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langgraph.graph import StateGraph, END
from datasets import load_dataset
import re
import json
import os
import tqdm
import argparse

# Parse arguments first
parser = argparse.ArgumentParser(description='Self-reflect on ARC failures using guided critiques')
parser.add_argument('--model', default=None, help='Model name or path to use for self-reflection')
parser.add_argument('--judge_file', default=None, help='Path to judge results file')
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

def load_gpt_judge_results(file_path):
    """Load GPT judge results to get failure mode mappings and original responses."""
    judge_results = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                judge_results[data['case_id']] = {
                    'failure_mode': data.get('failure_mode', []),
                    'mode_reasons': data.get('mode_reasons', []),
                    'model_response': data.get('model_response', '')
                }
    return judge_results

def generate_targeted_critique(failure_modes, mode_reasons, question, options, draft):
    """Generate targeted critique based on identified failure modes."""
    critiques = []

    for i, mode in enumerate(failure_modes):
        if mode == "NO_REASONING":
            critiques.append("• The answer lacks any reasoning or explanation. You should provide step-by-step thinking to justify your choice.")

        elif mode == "NO_REASONING_OUTLINE":
            critiques.append("• The reasoning lacks a clear outline or structured approach. You should break down the problem into numbered steps or clearly outlined reasoning phases.")

        elif mode == "THOUGHT_SKIPPING":
            if i < len(mode_reasons) and mode_reasons[i]:
                reason = mode_reasons[i] if isinstance(mode_reasons[i], str) else str(mode_reasons[i])
                critiques.append(f"• The reasoning skips important steps: {reason}. Make sure to complete each step of your planned approach before moving to the next.")
            else:
                critiques.append("• The reasoning appears to skip important intermediate steps. Make sure to complete each step of your planned approach before moving to the next.")

        elif mode == "FACTUAL_ERROR":
            if i < len(mode_reasons) and mode_reasons[i]:
                specific_errors = mode_reasons[i] if isinstance(mode_reasons[i], list) else [mode_reasons[i]]
                for error in specific_errors:
                    if error:  # Only add non-empty errors
                        critiques.append(f"• Factual error identified: {error}. Please verify and correct this information.")

        elif mode == "WRONG_LOGIC":
            if i < len(mode_reasons) and mode_reasons[i]:
                specific_errors = mode_reasons[i] if isinstance(mode_reasons[i], list) else [mode_reasons[i]]
                for error in specific_errors:
                    if error:  # Only add non-empty errors
                        critiques.append(f"• Logical error identified: {error}. Please reconsider this reasoning step.")

    # If no specific critiques were generated, the response is good as-is
    if not critiques:
        return None  # No critiques needed

    return "\n".join(critiques)


revise_prompt = """Revise the draft answer using the critiques. Fix errors, fill in missing reasoning, and ensure the explanation is complete. Finally return your the letter or number of the option as your answer like `The answer is {{the letter or number of the option}}`

Question: {question}
Options: {options}
Draft Answer: {draft}
Critiques: {critiques}

Revised Answer:"""

def load_draft(state):
    """Load the original model response as draft from judge results."""
    question_id = state.get('question_id')
    if question_id and question_id in judge_results:
        draft = judge_results[question_id]['model_response']
    else:
        # Fallback: generate new response if not found in judge results
        draft = "No original response found."
    return {**state, "draft": draft}

def critique_with_failure_modes(state):
    # Get targeted critiques based on failure modes
    question_id = state.get('question_id')
    if question_id and question_id in judge_results:
        failure_modes = judge_results[question_id]['failure_mode']
        mode_reasons = judge_results[question_id]['mode_reasons']
        targeted_critiques = generate_targeted_critique(
            failure_modes, mode_reasons,
            state['question'], state['options'], state['draft']
        )

        # If no critiques needed, skip critique phase
        if targeted_critiques is None:
            return {**state, "critiques": None, "targeted_critiques": "No issues identified - response is good as-is"}
    else:
        # No judge results available, assume response needs review
        targeted_critiques = "• Please provide more detailed reasoning to support your answer choice."

    # Use targeted critiques directly as the critiques
    return {**state, "critiques": targeted_critiques, "targeted_critiques": targeted_critiques}

def revise(state):
    # If no critiques were generated, use the original draft as final
    if state.get("critiques") is None:
        return {**state, "final": state["draft"]}

    prompt = revise_prompt.format(**state)
    final = llm.invoke(prompt)
    return {**state, "final": final}

# Load GPT judge results for failure mode guidance
judge_file_path = args.judge_file
judge_results = load_gpt_judge_results(judge_file_path)
print(f"Loaded {len(judge_results)} judge results for failure mode guidance")

# Create the self-reflection graph
g = StateGraph(dict)
g.add_node("load_draft", load_draft)
g.add_node("critique", critique_with_failure_modes)
g.add_node("revise", revise)
g.set_entry_point("load_draft")
g.add_edge("load_draft", "critique")
g.add_edge("critique", "revise")
g.add_edge("revise", END)
app = g.compile()

# Load ARC dataset
raw_dataset = load_dataset(path="allenai/ai2_arc", name="ARC-Challenge")
eval_dataset = raw_dataset["test"]
save_path = args.output

test_queries = []

if os.path.exists(save_path):
    with open(save_path, "r") as f:
        for line in f:
            if line.strip():
                test_queries.append(json.loads(line))
    answered_questions = set([q["id"] for q in test_queries])
    eval_dataset = [q for q in eval_dataset if q["id"] not in answered_questions]
    print(f"Skipping {len(test_queries)} already answered questions.")

# Process each question with failure-mode-guided self-reflection
for query in tqdm.tqdm(eval_dataset):
    question = query["question"]
    choices = query["choices"]
    question_id = query["id"]

    options = "\n".join([choices['label'][i] + ". " + choices['text'][i] for i in range(len(choices['label']))])

    # Run self-reflection on the question with failure mode guidance
    state = {
        "question": question,
        "options": options,
        "question_id": question_id
    }

    result = app.invoke(state)

    # Extract answer from the final response
    final_response = result["final"]
    # match = re.search(r"answer is ([A-Za-z0-9])", final_response)

    # if match:
    #     answer = match.group(1).strip()
    # else:
    #     answer = "N/A"

    # # Check accuracy
    # if answer == query["answerKey"]:
    #     query["acc"] = 1
    # else:
    #     query["acc"] = 0

    # Store all reflection steps including targeted critiques
    query["draft"] = result["draft"]
    query["targeted_critiques"] = result["critiques"]
    query["final_response"] = final_response
    # query["predicted_answer"] = answer

    # Add failure mode information if available
    if question_id in judge_results:
        query["failure_modes_used"] = judge_results[question_id]['failure_mode']
        query["mode_reasons_used"] = judge_results[question_id]['mode_reasons']

    # Save to file
    with open(save_path, "a") as f:
        f.write(json.dumps(query))
        f.write("\n")

    # print(f"Question: {question[:100]}...")
    # print(f"Correct: {query['answerKey']}, Predicted: {answer}, Accuracy: {query['acc']}")
    # print("-" * 80)