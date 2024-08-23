<div align="center"><h1>&nbsp;PEARL: Parallel Speculative Decoding with Adaptive Draft Length</h1></div>

<p align="center">
| <a href="https://arxiv.org/pdf/2408.11850"><b>Paper </b></a> | 
<a href="https://pearl-code.github.io/"><b>Blog</b></a> |
</p>
---

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://s2.loli.net/2024/08/13/u3tc4FAwxQG126y.png" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Figure 1. Speedup on HumanEval. All the experiments are conducted with H100 80G GPUs. The part results of <a href="https://github.com/thunlp/Ouroboros">Ouroboros</a> and <a href="https://github.com/hao-ai-lab/LookaheadDecoding">Lookahead Decoding</a> are reproduced with their official codes.
  	</div>
</center>



<br>

> TL; DR: we introduce **PEARL** (Parallel spEculative decoding with Adaptive dRaft Length) to further reduce the inference latency of Large Language Models (LLMs). PEARL is a **parallel** inference framework based on [speculative decoding](https://arxiv.org/abs/2211.17192) which utilizes *pre-verify* and *post-verify* to achieve adaptive draft length. In summary, our PEARL is:
>
> - &#128293; up to **3.87**$\times$, **3.81**$\times$, **3.59**$\times$ and **3.95**$\times$ on HumanEval, GSM8K, MT-bench and MGSM, respectively.
> - **provably lossless**
> - **training-free**, and does not need additional memory
> - &#128293; can be applied to any algorithms based on draft-then-verify framework, such as [EAGLE](https://sites.google.com/view/eagle-llm) and [Medusa](https://sites.google.com/view/medusa-llm)
> - &#128293; Eliminating the burden of searching the optimal draft length, together with a larger expectation of accepted tokens.

<br>

<!-- Using HTML to center the abstract -->



---

<br>

<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Demo</h2>
        <div class="content has-text-justified">
        </div>
    </div>
</div>

![AR-demo](static/AR-demo.gif)

<p align="center" style="color:gray;">Figure 2.  Generation speed of Llama 2 chat 70B using PEARL and auto-regressive decoding, with inference conducted on A100 80G GPUs at bf16 precision. </p>

---

<br>

<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Overview of PEARL</h2>
        <div class="content has-text-justified">
        </div>
    </div>
</div>


Our PEARL framework consists of a draft model, a target model and two strategies to decode tokens. The two strategies are switched according to the verification results in the last decoding step.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://s2.loli.net/2024/08/13/aoCAybN7S2KWsXd.png" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Figure 3. Overview of PEARL. PEARL achieves parallelism through adaptively using pre-verify and post-verify.
  	</div>
</center>


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

## With UI
We have provided a suggested web interface, which you can use by running the following command. 
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 applications --eval_mode para_sd --gamma 5 -n 1  -e applications --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0
```


<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Citation</h2>
        <div class="content has-text-justified">
        </div>
    </div>
</div>


If you find our work useful your research, please cite our paper:

```
@misc{liu2024parallelspeculativedecodingadaptive,
      title={Parallel Speculative Decoding with Adaptive Draft Length}, 
      author={Tianyu Liu and Yun Li and Qitan Lv and Kai Liu and Jianchen Zhu and Winston Hu},
      year={2024},
      eprint={2408.11850},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.11850}, 
}
```