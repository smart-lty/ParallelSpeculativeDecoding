# 7b models auto-regressive decoding

# gsm8k
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_gsm8k.py --eval_mode small -n 1  -e G_AR_Llama2_7b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 512 --temp 0 

# humaneval
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode small -n 1  -e H_AR_deepseek_1.3b --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0 
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode small -n 1  -e H_AR_deepseek_6.7b --draft_model deepseek-6.7b --target_model deepseek-33b --max_tokens 1024 --temp 0 
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode small -n 1  -e H_AR_codellama_7b --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0 
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode small -n 1  -e H_AR_Llama2_7b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 1024 --temp 0 

# mt-bench
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_mt_bench.py --eval_mode small -n 1  -e M_AR_Llama2_7b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 256 --temp 0


# 33/34b models auto-regressive decoding

# humaneval
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode large -n 1  -e H_AR_deepseek_33b --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0 
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode large -n 1  -e H_AR_codellama_34b --draft_model codellama-7b --target_model codellama-34b --max_tokens 1024 --temp 0 


# 70b models auto-regressive decoding

# gsm8k
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 benchmark/eval_gsm8k.py --eval_mode large -n 1  -e test --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 512 --temp 0 

# humaneval
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode large -n 1  -e H_AR_codellama_70b --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0 
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode large -n 1  -e H_AR_Llama2_70b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 1024 --temp 0

# mt-bench
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 benchmark/eval_mt_bench.py --eval_mode large -n 1  -e M_AR_Llama2_70b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 256 --temp 0

