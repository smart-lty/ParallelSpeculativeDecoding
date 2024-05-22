import os
import sys
sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import ipdb
import time
import random
import shortuuid
from src.util import seed_everything, parse_arguments
from src.engine import Decoding
from fastchat.model import get_conversation_template
from Ouroboros.ouroboros.ouroboros import ouroboros
from Ouroboros.ouroboros.models import LlamaForCausalLM


def read_results(file_path):
    f = open(file_path)
    data = [json.loads(line) for line in f.readlines()]
    record = {}
    for item in data:
        if item["category"] not in record:
            record[item["category"]] = {"wall_time":[], "num_token": []}
        for choice in item["choices"]:
            record[item["category"]]["wall_time"].extend(choice["wall_time"])
            record[item["category"]]["num_token"].extend(choice["num_token"])
    speed_record = {}
    for k in record:
        if k == "writing":
            speed = sum(record[k]["num_token"][1:]) / sum(record[k]["wall_time"][1:])
        else:
            speed = sum(record[k]["num_token"]) / sum(record[k]["wall_time"])
        speed_record[k] = speed
    return speed_record


class EvalMTBench(Decoding):
    def __init__(self, args):
        super().__init__(args)

        # load relative resources
        self.load_tokenizer()
        self.load_data()
        self.load_model()

        if "Llama-2" in self.args.draft_model and "Llama-2" in self.args.target_model:
            self.model_id = "llama-2-chat"
        elif "vicuna" in self.args.draft_model and "vicuna" in self.args.target_model:
            self.model_id = "vicuna"
        else:
            raise NotImplementedError

    def load_model(self):
        # * load models according to different evaluation methods.
        self.color_print(f"Loading models:\n{self.args.draft_model}\n{self.args.target_model}", 3)
        self.draft_model = LlamaForCausalLM.from_pretrained(self.args.draft_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        self.target_model = LlamaForCausalLM.from_pretrained(self.args.target_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        self.window_size = 20
        self.guess_set_size = 20
        self.lookahead_level = 7
        self.gamma = 12

    def load_data(self):
        # * load evaluation data
        self.color_print(f"Loading MT-bench data...", 3)
        data = []
        with open(os.path.join(self.args.data_path, "mt_bench.jsonl")) as f:
            for line in f.readlines():
                datum = json.loads(line)
                data.append(datum)
        self.data = data

    def preprocess(self, input_text):
        pass
    
    def postprocess(self, input_text, output_text):
        pass

    @torch.no_grad()
    def eval(self):
        if self.args.eval_mode == "small" or self.args.eval_mode == "large":
            decoding = self.autoregressive_sampling
        elif self.args.eval_mode == "sd":
            decoding = self.speculative_decoding
        elif self.args.eval_mode == "para_sd":
            decoding = self.parallel_speculative_decoding
        else:
            raise NotImplementedError
        
        out_path = os.path.join(self.args.exp_name, f"{self.args.eval_mode}_mt_bench.jsonl")
        out_f = open(out_path, "a")
        
        for question in tqdm.tqdm(self.data, total=len(self.data), disable=not self.accelerator.is_main_process, ncols=50):
            
            choices = []
            # set random seed. Ensure each experiment runs with a unique random seed.
            for i in range(self.args.num_samples_per_task):
                while self.seed in self.seed_set:
                    self.seed = random.randint(0, 1000000)
                seed_everything(self.seed)

                conv = get_conversation_template(self.model_id)
                if self.model_id == "llama-2-chat":
                    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
                    conv.system_message = sys_p
                turns = []
                wall_time = []
                num_token = []
                for turn_idx in range(len(question["turns"])):
                    qs = question["turns"][turn_idx]
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt() + " "
                    input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids = ouroboros(input_ids, self.draft_model, self.target_model, max_len=self.args.max_tokens, gamma=self.gamma, window_size=self.window_size, guess_set_size=self.guess_set_size, lookahead_level=self.lookahead_level)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    output_text = self.tokenizer.decode(output_ids[0], spaces_between_special_tokens=False)
                    
                    for special_token in self.tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output_text = output_text.replace(special_tok, "")
                        else:
                            output_text = output_text.replace(special_token, "")
                    output_text = output_text.strip()
                    conv.messages[-1][-1] = output_text
                    turns.append(output_text)
                    wall_time.append(end_time - start_time)
                    num_token.append(output_ids.shape[1] - input_ids.shape[1])
                choices.append({"index": i, "wall_time": wall_time, "num_token": num_token, "turns": turns})

            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": self.model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            if self.accelerator.is_main_process:
                out_f.write(json.dumps(ans_json, ensure_ascii=False) + "\n")
                out_f.flush()
        out_f.close()

        self.color_print(f"current eval mode: {self.args.eval_mode}", 0)
        speed = read_results(out_path)
        for k, v in speed.items():
            self.color_print(f"Generating speed of category {k}: {v:.2f} token / second", 2)
        self.color_print(f"Average generating speed: {sum(speed.values())/len(speed.values())} token / second", 2)
        self.color_print(f"draft model forward times: {self.draft_forward_times}", 2)
        
        self.accelerator.wait_for_everyone()
        
        if (self.accelerator.num_processes == 1 and self.accelerator.is_main_process) or (self.accelerator.num_processes == 2 and not self.accelerator.is_main_process):
            print(f"\033[92mtarget model forward times: {self.target_forward_times}\033[0m")
        
        self.accelerator.wait_for_everyone()

if __name__ == "__main__":
    args = parse_arguments()
    alg = EvalMTBench(args)
    alg.eval()
    