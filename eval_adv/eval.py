import argparse
import glob
import json
import os
import random
from collections import defaultdict

import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM, AutoConfig


from utils import create_prompt_with_llama2_chat_format, gpt_eval, asr_eval, generate_completions
import pandas as pd



def get_prompt(instruction, examples = []):
    exp_str = ""
    for e in examples:
        exp_str+=f"\n\n### Instruction:\n{e[0]}\n\n### Response:\n{e[1]}"

    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.{exp_str}\n\n### Instruction:\n{instruction}\n\n### Response:"
    return prompt


def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    all_questions = list(pd.read_csv("harmful_behaviors.csv")["goal"])
    all_targets = list(pd.read_csv("harmful_behaviors.csv")["target"])
    assert args.n_shots + args.eval_examples <= len(all_questions)
    questions = all_questions[:args.eval_examples]

    # select examples from back
    examples = []
    if args.n_shots > 0:
        for i in range(args.n_shots):
            examples.append((all_questions[-i-1], all_targets[-i-1]))
    
    


    prompts = []
    for data in questions:
        prompt = get_prompt(data, examples=examples)
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = create_prompt_with_llama2_chat_format(messages, add_bos=False)
        prompts.append(prompt)

    print(prompts[0], flush=True)

    model_kwargs = {
        'device_map': "auto",
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
        'load_in_8bit': args.load_in_8bit,
    }
    if args.revision:
        model_kwargs['revision'] = args.revision

    if args.subfolder:
        model_kwargs['subfolder'] = args.subfolder

    if args.config_name:
        model_kwargs['config'] = AutoConfig.from_pretrained(args.config_name)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.eval()


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=True)

    # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # for OPT and Pythia models, we need to set tokenizer.model_max_length to model.config.max_position_embeddings
    # to avoid wrong embedding index.
    if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
        tokenizer.model_max_length = model.config.max_position_embeddings
        print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))


    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=512,
        batch_size=args.eval_batch_size if args.eval_batch_size else 1,
        do_sample=False,
    )


    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for i in range(len(questions)):
            example = {"instruction": questions[i], "response": outputs[i]}
            fout.write(json.dumps(example) + "\n")

    if args.eval:
        if args.metric=="gpt4":
            metric_model = "gpt-4-1106-preview"
        elif args.metric=="gpt3.5":
            metric_model = "gpt-3.5-turbo-0125"
        elif  args.metric=="gpt4o":
            metric_model = "gpt-4o-2024-08-06"
        elif  args.metric=="gpt4omini":
            metric_model = "gpt-4o-mini-2024-07-18"
        else:
            raise ValueError
        kw_asr = asr_eval(outputs=outputs)
        gpt_score, gpt_asr = gpt_eval(instructions=questions, outputs=outputs, cache_path=os.path.join(args.save_dir, f"{args.metric}_eval_cache.jsonl"), model=metric_model)
        performance = {"samples": len(questions), "kw_asr": kw_asr, "gpt_asr": gpt_asr, "gpt_score": gpt_score}
        print(performance)
        with open(os.path.join(args.save_dir, f"{args.metric}_metrics.jsonl"), "w") as fout:
            fout.write(json.dumps(performance, indent=4)+"\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/toxigen"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--eval_examples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--eval",
        action="store_true",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default='gpt4o',
    )
    args = parser.parse_args()

    # assert args.model_name_or_path or args.base_model_name_or_path

    main(args)
