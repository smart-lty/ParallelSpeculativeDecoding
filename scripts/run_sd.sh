
# 7b & 34b parallel speculative decoding
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode sd --gamma 8 -n 1  -e test1 --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode sd --gamma 6 -n 1  -e test2 --draft_model deepseek-6.7b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode sd --gamma 6 -n 1  -e test3 --draft_model codellama-7b --target_model codellama-34b --max_tokens 1024 --temp 0

# 7b & 70b parallel speculative decoding
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 benchmark/eval_gsm8k.py --eval_mode sd --gamma 10 -n 1  -e G_SD_Llama2_7_70b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 512 --temp 0 
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode sd --gamma 10 -n 1  -e H_SD_codellama_7_70b --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0 
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode sd --gamma 10 -n 1  -e H_SD_Llama2_7_70b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 benchmark/eval_mt_bench.py --eval_mode sd --gamma 10 -n 1  -e M_SD_Llama2_7_70b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 512 --temp 0
