CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 comparation/eval_humaneval_lookahead.py --eval_mode sd -n 1  -e lookahead_deepseek_1.3_33b --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 comparation/eval_humaneval_lookahead.py --eval_mode sd -n 1  -e lookahead_deepseek_6.7_33b --draft_model deepseek-6.7b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 comparation/eval_humaneval_lookahead.py --eval_mode sd -n 1  -e lookahead_codellama_7_34b --draft_model codellama-7b --target_model codellama-34b --max_tokens 1024 --temp 0

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 comparation/eval_humaneval_ouroboros.py --eval_mode sd -n 1  -e ouroboros_deepseek_1.3_33b --draft_model deepseek-1.3b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 comparation/eval_humaneval_ouroboros.py --eval_mode sd -n 1  -e ouroboros_deepseek_6.7_33b --draft_model deepseek-6.7b --target_model deepseek-33b --max_tokens 1024 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1 comparation/eval_humaneval_ouroboros.py --eval_mode sd -n 1  -e ouroboros_codellama_7_34b --draft_model codellama-7b --target_model codellama-34b --max_tokens 1024 --temp 0


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 comparation/eval_humaneval_lookahead.py --eval_mode sd -n 1  -e lookahead_codellama_7_70b --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0 
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 comparation/eval_humaneval_lookahead.py --eval_mode sd -n 1  -e lookahead_Llama2_7_70b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 1024 --temp 0

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 comparation/eval_gsm8k_lookahead.py --eval_mode sd -n 1  -e G_lookahead_Llama2_7_70b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 256 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 comparation/eval_mt_bench_ouroboros.py --eval_mode sd -n 1  -e M_ouroboros_Llama2_7_70b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 256 --temp 0
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 comparation/eval_humaneval_ouroboros.py --eval_mode sd -n 1  -e ouroboros_codellama_7_70b --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0 
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 comparation/eval_humaneval_ouroboros.py --eval_mode sd -n 1  -e ouroboros_Llama2_7_70b --draft_model llama-2-7b --target_model llama-2-70b --max_tokens 1024 --temp 0

