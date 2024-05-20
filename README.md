# Document for evaluation and reproduction

This is the code for paper "Parallel Speculative Decoding with Adaptive Draft Length".

## File Tree

- `src`: the core codes to implement auto-regressive decoding, vanilla speculative decoding and parallel speculative decoding.
- `benchmark`: evaluation files, to evaluate the decoding algorithms in specific tasks.
- `comparison`: files to reproduce the results of compared baselines in the paper.
- `scripts`: scripts for all experiments in the paper.
- `data`: data file.
- `install.sh`: helper for installing relevant packages.



## preparation

Follow the instructions below to prepare for reproducing the results in the paper.

1. experimental environment: `sh install.sh` will install the necessary packages in the project.
2. code changes: changes the code `src/util.py` line 31-38 and line 49, to fill in your model paths and data paths.



## reproduction

All the running scripts, including scripts for auto-regress decoding, vanilla speculative decoding, parallel speculative decoding, comparison, ablation studies and case studies. These scripts can be directly executed for reproduction.

```shell
sh scripts/run_para_sd.sh
```



## Examples

You can try this code with a simple command:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 5 -n 1  -e H_PSD_codellama_7_70b --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0
```

