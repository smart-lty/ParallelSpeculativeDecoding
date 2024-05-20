# * Part 1: optimal gamma
## ? Codellama 7&34b, 7&70b, deepseek 1.3&33b, 7&33b on HumanEval
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 2 -n 1  -e H_PSD_codellama_7_34b_gamma_2 --draft_model codellama-7b --target_model codellama-34b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 4 -n 1  -e H_PSD_codellama_7_34b_gamma_4 --draft_model codellama-7b --target_model codellama-34b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 5 -n 1  -e H_PSD_codellama_7_34b_gamma_5 --draft_model codellama-7b --target_model codellama-34b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 6 -n 1  -e H_PSD_codellama_7_34b_gamma_6 --draft_model codellama-7b --target_model codellama-34b --max_tokens 1024 --temp 0 

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 2 -n 1  -e H_PSD_codellama_7_70b_gamma_2 --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 3 -n 1  -e H_PSD_codellama_7_70b_gamma_3 --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 4 -n 1  -e H_PSD_codellama_7_70b_gamma_4 --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 6 -n 1  -e H_PSD_codellama_7_70b_gamma_6 --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0 

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 2 -n 1  -e H_PSD_deepseek_1.3_33b_gamma_2 --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 3 -n 1  -e H_PSD_deepseek_1.3_33b_gamma_3 --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 5 -n 1  -e H_PSD_deepseek_1.3_33b_gamma_5 --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 6 -n 1  -e H_PSD_deepseek_1.3_33b_gamma_6 --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 2 -n 1  -e H_PSD_deepseek_6.7_33b_gamma_2 --draft_model deepseek-6.7b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 4 -n 1  -e H_PSD_deepseek_6.7_33b_gamma_4 --draft_model deepseek-6.7b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 5 -n 1  -e H_PSD_deepseek_6.7_33b_gamma_5 --draft_model deepseek-6.7b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 6 -n 1  -e H_PSD_deepseek_6.7_33b_gamma_6 --draft_model deepseek-6.7b --target_model deepseek-33b --max_tokens 1024 --temp 0

## ? Llama 2 7&70b on HumanEval, GSM8K and MT-bench
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 3 -n 1  -e H_PSD_Llama2_7_70b_gamma_3 --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 4 -n 1  -e H_PSD_Llama2_7_70b_gamma_4 --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 6 -n 1  -e H_PSD_Llama2_7_70b_gamma_6 --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 7 -n 1  -e H_PSD_Llama2_7_70b_gamma_7 --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 1024 --temp 0

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd --gamma 3 -n 1  -e G_PSD_Llama2_7_70b_gamma_3 --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 256 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd --gamma 4 -n 1  -e G_PSD_Llama2_7_70b_gamma_4 --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 256 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd --gamma 6 -n 1  -e G_PSD_Llama2_7_70b_gamma_6 --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 256 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd --gamma 7 -n 1  -e G_PSD_Llama2_7_70b_gamma_7 --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 256 --temp 0

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_mt_bench.py --eval_mode para_sd --gamma 3 -n 1  -e M_PSD_Llama2_7_70b_gamma_3 --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 256 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_mt_bench.py --eval_mode para_sd --gamma 4 -n 1  -e M_PSD_Llama2_7_70b_gamma_4 --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 256 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_mt_bench.py --eval_mode para_sd --gamma 6 -n 1  -e M_PSD_Llama2_7_70b_gamma_6 --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 256 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_mt_bench.py --eval_mode para_sd --gamma 7 -n 1  -e M_PSD_Llama2_7_70b_gamma_7 --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 256 --temp 0

# * Part 2: mean accepted tokens

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode sd --gamma 6 -n 1  -e H_SD_codellama_7_34b_gamma_6 --draft_model codellama-7b --target_model codellama-34b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode sd --gamma 10 -n 1  -e H_SD_codellama_7_70b_gamma_10 --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode sd --gamma 8 -n 1  -e H_SD_deepseek_1.3_33b_gamma_8 --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode sd --gamma 6 -n 1  -e H_SD_deepseek_6.7_33b_gamma_6 --draft_model deepseek-6.7b --target_model deepseek-33b --max_tokens 1024 --temp 0

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 3 -n 1  -e H_PSD_codellama_7_34b_gamma_3 --draft_model codellama-7b --target_model codellama-34b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 5 -n 1  -e H_PSD_codellama_7_70b_gamma_5 --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 4 -n 1  -e H_PSD_deepseek_1.3_33b_gamma_4 --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 3 -n 1  -e H_PSD_deepseek_6.7_33b_gamma_3 --draft_model deepseek-6.7b --target_model deepseek-33b --max_tokens 1024 --temp 0
