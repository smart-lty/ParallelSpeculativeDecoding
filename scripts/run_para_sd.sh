# 7b & 34b parallel speculative decoding
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 4 -n 1  -e H_PSD_deepseek_1.3_33b --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 3 -n 1  -e H_PSD_deepseek_6.7_33b --draft_model deepseek-6.7b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 3 -n 1  -e H_PSD_codellama_7_34b --draft_model codellama-7b --target_model codellama-34b --max_tokens 1024 --temp 0

# 7b & 70b parallel speculative decoding
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd --gamma 5 -n 1 -e G_PSD_llama2_7_70b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 512 --temp 0 
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 5 -n 1  -e H_PSD_codellama_7_70b --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0 
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 5 -n 1  -e H_PSD_Llama2_7_70b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_mt_bench.py --eval_mode para_sd --gamma 5 -n 1  -e M_PSD_Llama2_7_70b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 512 --temp 0
