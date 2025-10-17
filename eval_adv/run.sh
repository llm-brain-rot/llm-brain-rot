export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m eval \
--eval_examples 100 \
--n_shots 0 \
--save_dir results/base_models/Llama-3.1-8B \
--model_name_or_path meta-llama/Llama-3.1-8B \
--eval \
--metric gpt4o \
--eval_batch_size 1 