import os
import sys
sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import time
import ipdb
import random
from src.util import seed_everything, parse_arguments
from src.engine import Decoding
from transformers import AutoModelForCausalLM, AutoTokenizer


class EvalHumaneval(Decoding):
    def __init__(self, args):
        super().__init__(args)
        
        # load relative resources
        self.load_tokenizer()
        self.load_data()
        self.load_model()


    def load_model(self):
        # * load models according to different evaluation methods.
        self.color_print("Loading models...", 3)
        self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

        self.vocab_size = self.args.vocab_size
        self.draft_model.config.vocab_size = self.vocab_size
        self.target_model.config.vocab_size = self.vocab_size

    def load_data(self):
        # * load evaluation data
        self.color_print(f"Loading HumanEval data...", 3)
        data = []
        with open(os.path.join(self.args.data_path, "humaneval.jsonl")) as f:
            for line in f.readlines():
                datum = json.loads(line)
                datum["input_text"] = self.preprocess(datum["prompt"])
                encode_special_token_flag = not ("Llama-3.1" in self.args.draft_model and "Llama-3.1" in self.args.target_model)
                input_ids = self.tokenizer.encode(datum["input_text"], add_special_tokens=encode_special_token_flag)
                datum["input_ids"] = torch.tensor(input_ids).unsqueeze(0)
                data.append(datum)
        self.data = data

    def preprocess(self, input_text):
        text = input_text.strip()
        return text

    def postprocess(self, input_text, output_text):
        if output_text.startswith(self.tokenizer.bos_token):
            generation = output_text[len(input_text)+len(self.tokenizer.bos_token)+1:] # tokenizer will add a '<s> ' at the beginning of the text.
        else:
            generation = output_text[len(input_text):]
        stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", self.tokenizer.eos_token]
        for stop_word in stop_words:
            if stop_word in generation:
                next_line = generation.index(stop_word)
                generation = generation[:next_line].strip()
        output_text = input_text + '\n    ' + generation
        output_text = output_text.replace("\t", "    ")
        
        return output_text
             
    @torch.no_grad()
    def eval(self):
        out_path = os.path.join(self.args.exp_name, f"{self.args.eval_mode}_humaneval.jsonl")
        out_f = open(out_path, "a")
        wall_times = {"time":[], "num_tokens":[]}
        for _ in range(self.args.num_samples_per_task):
            # set random seed. Ensure each experiment runs with a unique random seed.
            while self.seed in self.seed_set:
                self.seed = random.randint(0, 1000000)
            seed_everything(self.seed)
            self.seed_set.add(self.seed)

            for datum in tqdm.tqdm(self.data, total=len(self.data), disable=not self.accelerator.is_main_process, ncols=50):
                input_ids = datum["input_ids"]
                torch.cuda.synchronize()
                start_time = time.time()
                generate_ids = self.target_model.generate(input_ids.cuda(), max_new_tokens=self.args.max_tokens, temperature=self.args.temp, do_sample=False, assistant_model=self.draft_model)
                torch.cuda.synchronize()
                end_time = time.time()
                if self.accelerator.is_main_process:
                    if datum["task_id"] != "HumanEval/0":
                        # skip the first prompt time consumption
                        wall_times["time"].append(end_time-start_time)
                        wall_times["num_tokens"].append(generate_ids.shape[1] - input_ids.shape[1])
                    output = self.postprocess(datum["input_text"], self.tokenizer.decode(generate_ids[0, :]))
                    out_f.write(json.dumps({"task_id": datum["task_id"], "completion": output}, ensure_ascii=False) + "\n")
                out_f.flush()
        
        out_f.close()
        
        if self.accelerator.is_main_process:
            speed = sum(wall_times["num_tokens"]) / sum(wall_times["time"])
            self.color_print(f"generate speed (tokens / second): {speed:.2f}", 2)


if __name__ == "__main__":
    args = parse_arguments()
    alg = EvalHumaneval(args)
    alg.eval()