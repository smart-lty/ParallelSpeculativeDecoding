## ? Codellama 7&34b, 7&70b, deepseek 1.3&33b on HumanEval, Llama 2 7&70b on GSM8K, without strategy 1 and without strategy 2

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd_wo_1 --gamma 3 -n 1  -e abl_G_PSD_codellama_7_34b_wo_1 --draft_model codellama-7b --target_model codellama-34b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd_wo_2 --gamma 3 -n 1  -e abl_G_PSD_codellama_7_34b_wo_2 --draft_model codellama-7b --target_model codellama-34b --max_tokens 1024 --temp 0

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd_wo_1 --gamma 5 -n 1  -e abl_H_PSD_codellama_7_70b_wo_1 --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd_wo_2 --gamma 5 -n 1  -e abl_H_PSD_codellama_7_70b_wo_2 --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd_wo_1 --gamma 4 -n 1  -e abl_H_PSD_deepseek_1.3_33b_wo_1 --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd_wo_2 --gamma 4 -n 1  -e abl_H_PSD_deepseek_1.3_33b_wo_2 --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd_wo_1 --gamma 5 -n 1  -e abl_G_PSD_Llama2_7_70b_wo_1 --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 256 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_gsm8k.py --eval_mode para_sd_wo_2 --gamma 5 -n 1  -e abl_G_PSD_Llama2_7_70b_wo_2 --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 256 --temp 0
