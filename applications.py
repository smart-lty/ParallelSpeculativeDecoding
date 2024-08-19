import os
import torch
import time
import re
import gradio as gr
from fastchat.model import get_conversation_template
from accelerate import Accelerator
import accelerate
from src.kvcache import KVCacheModel
from src.util import sample, max_fn, parse_arguments, norm_logits
from transformers import AutoModelForCausalLM, AutoTokenizer

PADDING_LENGTH = int(1e6)

args = parse_arguments()
accelerator = Accelerator()
tokenizer = AutoTokenizer.from_pretrained(args.draft_model, trust_remote_code=True)
tokenizer.padding_side = "right"
tokenizer.pad_token_id = 2

if accelerator.num_processes == 2:
    if accelerator.is_main_process:
        model = AutoModelForCausalLM.from_pretrained(args.draft_model, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.target_model, device_map="balanced_low_0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(args.target_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

stop_token_list = [tokenizer.eos_token_id]
if "Llama-3.1" in args.draft_model:
    stop_token_list.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))


accelerator.wait_for_everyone()

@torch.no_grad()
def parallel_speculative_decoding(prefix):
    # parallel speculative decoding
    kv_model = KVCacheModel(model, args.temp, args.top_k, args.top_p)
    
    vocab_size = args.vocab_size
    kv_model.vocab_size = vocab_size
    device = kv_model._model.device
    max_tokens = prefix.shape[1] + args.max_tokens
    
    # this flag is used to determine the current verify mode.
    cur_mode = True
    num_acc_token = 0
    input_len = prefix.shape[1]

    while prefix.shape[1] < max_tokens:
        prefix_len = prefix.shape[1]
        
        input_ids = prefix.to(device)
        if accelerator.is_main_process:
            x = kv_model.generate(input_ids, args.gamma)
            prob = kv_model._prob_history[:, prefix_len-args.gamma-1:prefix_len, :vocab_size]
            prob[:, 0, 0] = -1
            prob[:, 0, 1:args.gamma*2] = x[:, prefix_len-args.gamma+1:prefix_len+args.gamma]
        else:
            x = kv_model.generate(input_ids, 1)
            prob = kv_model._prob_history[:, prefix_len-args.gamma-1:prefix_len, :vocab_size]
        
        accelerator.wait_for_everyone()
        all_prob = accelerator.gather(prob)
        draft_ids = all_prob[0, [0], 1:args.gamma*2].int()
        draft_prob = all_prob[[0], 1:, :]
        target_prob = all_prob[[1], 1:, :]
        
        if cur_mode:
            first_token = draft_ids[:, -args.gamma]
            torch.manual_seed(args.seed + prefix_len)

            r = torch.rand(1, device=device)
            if  r > target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]:
                # reject the first token
                t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                prefix = torch.cat((input_ids, t), dim=1)
                
                # record the number of accepted tokens
                num_acc_token = 0
                
                if accelerator.is_main_process:
                    # rollback the small model kv cache
                    kv_model.rollback(prefix_len)
            else:
                # accept the first token, change the mode
                cur_mode = False
                prefix = torch.cat((input_ids, draft_ids[:, -args.gamma:]), dim=1)
                num_acc_token += 1

        else:
            n = args.gamma
            for i in range(args.gamma):
                token = draft_ids[:, i]
                torch.manual_seed(args.seed + prefix_len - args.gamma + i)
                r = torch.rand(1, device=device)
                if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                    n = i
                    break
            if n == args.gamma:
                # accept all guess tokens
                prefix = torch.cat((input_ids, draft_ids[:, -args.gamma:]), dim=1)
                num_acc_token += args.gamma
            else:
                # reject someone, change the mode
                assert n < args.gamma
                cur_mode = True
                t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))
                
                prefix = torch.cat((input_ids[:, :prefix_len-args.gamma + n + 1], t), dim=1)
                num_acc_token = 0
                # rollback both the large model and the small model kv cache
                kv_model.rollback(prefix_len - args.gamma +n+1)
        
        yield prefix, num_acc_token
        
        flag = False
        for token in stop_token_list:
            if token in prefix[0, input_len:].tolist():
                flag = True
                break
        if flag:
            break

@torch.no_grad()
def autoregressive_sampling(prefix):
    prefix = prefix.to(model.device)

    prefix_len = prefix.shape[1]
    max_tokens = prefix_len + args.max_tokens
    
    x = prefix
    past_key_values = None
    while x.shape[1] < max_tokens:
        if past_key_values:
            last_ids = x[:, -1]
            if last_ids.dim() == 1:
                last_ids = last_ids.unsqueeze(0)
            outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)
        else:
            outputs = model(x)

        last_p = norm_logits(outputs.logits[::, -1, :], args.temp, args.top_k, args.top_p)
        past_key_values = outputs.past_key_values
        idx_next = sample(last_p)
        x = torch.cat((x, idx_next), dim=1)
        yield x

def user(user_message, history, session_state):
    if history==None:
        history=[]
    pure_history = session_state.get("pure_history", [])
    pure_history += [[user_message, None]]
    session_state["pure_history"] = pure_history
    return "", history + [[user_message, None]], session_state

def send_and_receive(user_message, history, session_state):
    msg = None
    if accelerator.is_main_process:
        msg = [user_message, history, session_state]
        accelerator.wait_for_everyone()
        return user(user_message, history, session_state)
    else:
        accelerator.wait_for_everyone()
        msg = accelerate.utils.gather_object(msg)
        user_message, history, session_state = msg
        return user(user_message, history, session_state)


def truncate_list(lst, num):
    if num not in lst:
        return lst
    first_index = lst.index(num)
    return lst[:first_index + 1]

def find_list_markers(text):
    pattern = re.compile(r'(?m)(^\d+\.\s|\n)')
    matches = pattern.finditer(text)
    return [(match.start(), match.end()) for match in matches]

def checkin(pointer,start,marker):
    for b,e in marker:
        if b<=pointer<e:
            return True
        if b<=start<e:
            return True
    return False

def highlight_text(text, text_list, color="black"):
    pointer = 0
    result = ""
    markers=find_list_markers(text)
    for sub_text in text_list:
        start = text.find(sub_text, pointer)
        if start==-1:
            continue
        end = start + len(sub_text)

        if checkin(pointer,start,markers):
            result += text[pointer:start]
        else:
            result += f"<span style='color: {color};'>{text[pointer:start]}</span>"
        result += sub_text
        pointer = end
    if pointer < len(text):
        result += f"<span style='color: {color};'>{text[pointer:]}</span>"

    return result

def regenerate(history,session_state):
    if not history:
        return history, None,"0.00 tokens/s","0.00",session_state
    pure_history = session_state.get("pure_history", [])
    pure_history[-1][-1] = None
    session_state["pure_history"]=pure_history
    if len(history) > 1:  # Check if there's more than one entry in history (i.e., at least one bot response)
        new_history = history[:-1]  # Remove the last bot response
        last_user_message = history[-1][0]  # Get the last user message
        return new_history + [[last_user_message, None]], None,"0.00 tokens/s","0.00",session_state
    history[-1][1] = None
    return history, None,"0.00 tokens/s","0.00",session_state

def clear(history,session_state):
    pure_history = session_state.get("pure_history", [])
    pure_history = []
    session_state["pure_history"] = pure_history
    return [],"0.00 tokens/s","0.00",session_state

def bot(history, use_Infer, highlight_Infer, session_state):
    if not history:
        return history, "0.00 tokens/s", "0.00", session_state
    pure_history = session_state.get("pure_history", [])

    if "Llama-3.1" in args.draft_model:
        model_type = "llama-3"
    elif "Llama-2" in args.draft_model:
        model_type = "llama-2-chat"
    elif "vicuna" in args.draft_model:
        model_type = "vicuna"
    else:
        raise NotImplementedError(f"unknown model types of {args.draft_model}")

    conv = get_conversation_template(model_type)


    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p


    for query, response in pure_history:
        conv.append_message(conv.roles[0], query)
        if model_type == "llama-2-chat" and response:
            response = " " + response
        conv.append_message(conv.roles[1], response)

    prompt = conv.get_prompt() + " "

    input_ids = tokenizer([prompt]).input_ids
    input_ids = torch.tensor(input_ids).to(model.device)
    
    copy_ids = torch.zeros((1, PADDING_LENGTH), device=input_ids.device).int()
    copy_ids[:, :input_ids.shape[1]] = input_ids
    copy_ids[0, -1] = input_ids.shape[1]
    accelerator.wait_for_everyone()
    copy_ids = accelerator.gather(copy_ids)

    input_len = input_ids.shape[1]

    naive_text = []
    
    totaltime = 0
    start_time=time.time()
    
    total_ids = 0

    if use_Infer:
        for output_ids, num_acc_token in parallel_speculative_decoding(prefix=input_ids):
            totaltime += time.time() - start_time

            decode_ids = output_ids[0, input_len:].tolist()
            decode_ids = truncate_list(decode_ids, tokenizer.eos_token_id)
            text = tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False, clean_up_tokenization_spaces=True, )
            if not num_acc_token:
                total_ids += 1
                naive_text.append(tokenizer.decode(output_ids[0, -1], skip_special_tokens=True, spaces_between_special_tokens=False, clean_up_tokenization_spaces=True, ))

            colored_text = highlight_text(text, naive_text, "blue")
            if highlight_Infer:
                history[-1][1] = colored_text
            else:
                history[-1][1] = text
            pure_history[-1][1] = text
            session_state["pure_history"] = pure_history
            new_tokens = output_ids.shape[1] - input_len
            if total_ids == 0:
                yield history,f"{new_tokens/totaltime:.2f} tokens/s",f"{new_tokens:.2f}",session_state
            else:
                yield history,f"{new_tokens/totaltime:.2f} tokens/s",f"{new_tokens/total_ids:.2f}",session_state
            start_time = time.time()


    else:
        for output_ids in autoregressive_sampling(prefix=input_ids):
            totaltime += (time.time() - start_time)
            total_ids+=1
            decode_ids = output_ids[0, input_len:].tolist()
            decode_ids = truncate_list(decode_ids, tokenizer.eos_token_id)
            text = tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False, clean_up_tokenization_spaces=True, )

            history[-1][1] = text
            pure_history[-1][1] = text
            new_tokens = output_ids.shape[1] - input_len
            yield history,f"{new_tokens/totaltime:.2f} tokens/s",f"{new_tokens/total_ids:.2f}",session_state
            start_time = time.time()

custom_css = """
#speed textarea {
color: red;   
font-size: 30px; 
}"""

if accelerator.is_main_process:
    with gr.Blocks(css=custom_css) as demo:
        gs = gr.State({"pure_history": []})
        gr.Markdown('''## PEARL Chatbot''')

        with gr.Row():
            speed_box = gr.Textbox(label="Speed", elem_id="speed", interactive=False, value="0.00 tokens/s")
            compression_box = gr.Textbox(label="Compression Ratio", elem_id="speed", interactive=False, value="0.00")
        with gr.Row():
            with gr.Column():
                use_Infer = gr.Checkbox(label="Use PEARL", value=True)
                highlight_Infer = gr.Checkbox(label="Highlight the tokens generated by PEARL", value=True)
            temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="temperature", value=0)
            top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="top_p", value=0.95)

        chatbot = gr.Chatbot(height=600,show_label=False)
        args.temp = temperature.value
        args.top_p = top_p.value

        msg = gr.Textbox(label="Your input")

        with gr.Row():
            send_button = gr.Button("Send")
            stop_button = gr.Button("Stop")
            regenerate_button = gr.Button("Regenerate")
            clear_button = gr.Button("Clear")


        enter_event=msg.submit(user, [msg, chatbot, gs], [msg, chatbot, gs], queue=True).then(
            bot, [chatbot, use_Infer, highlight_Infer, gs], [chatbot, speed_box, compression_box, gs]
        )
        clear_button.click(clear, [chatbot,gs], [chatbot,speed_box,compression_box,gs], queue=True)

        send_event=send_button.click(user, [msg, chatbot,gs], [msg, chatbot,gs],queue=True).then(
            bot, [chatbot, use_Infer, highlight_Infer, gs], [chatbot, speed_box, compression_box, gs]
        )
        regenerate_event=regenerate_button.click(regenerate, [chatbot,gs], [chatbot, msg,speed_box,compression_box,gs],queue=True).then(
            bot, [chatbot, use_Infer, highlight_Infer, gs], [chatbot, speed_box, compression_box, gs]
        )
        stop_button.click(fn=None, inputs=None, outputs=None, cancels=[send_event,regenerate_event,enter_event])
        
        demo.queue()
        demo.launch(share=True)
else:
    while True:
        copy_ids = torch.zeros((1, PADDING_LENGTH), device=accelerator.device).int()
        accelerator.wait_for_everyone()
        copy_ids = accelerator.gather(copy_ids)
        input_ids = copy_ids[[0], :copy_ids[0, -1].int()]
        print(input_ids)
        for _ in parallel_speculative_decoding(input_ids):
            continue