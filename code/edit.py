import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Model
from util import nethook
from util.generate import generate_interactive, generate_fast
from demo import demo_model_editing, stop_execution, print_loud
import pdb
import numpy as np
import random

CUDA = "cuda:5"

def main():
    MODEL_NAME = "gpt2-xl" # "gpt-j-6b", "llama-2-7b"
    if 'llama' in MODEL_NAME.lower():
        model, tok = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(
            CUDA
        ),
        AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False),
        )
    else:
        model, tok = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(
            CUDA
        ),
        AutoTokenizer.from_pretrained(MODEL_NAME),
        )
    tok.pad_token = tok.eos_token
    if 'llama' in MODEL_NAME.lower():
        tok.padding_side = 'right'
    
    request = [
    {
        "prompt": "{} is a citizen of",
        "subject": "Lionel Messi",
        "target_new": {"str": "China"},
    }
    ]
    generation_prompts = [
        "Lionel Messi is a citizen of",
        "Cristiano Ronaldo is a citizen of",
        "Lionel Messi plays for the club called"
    ]
    
    # Execute rewrite
    ALG_NAME = "RETS"
    model_new, orig_weights = demo_model_editing(
        model, tok, request, generation_prompts, alg_name=ALG_NAME
    )
    
    ##save model
    # model_new.save_pretrained(SAVE_MODEL_PATH)
    # tok.save_pretrained(SAVE_MODEL_PATH)
    
    generate_interactive(model_new, tok, max_out_len=200, use_logit_lens=True)#For llama, add(..., ln_f_module="model.norm", layer_module_tmp="model.layers.{}", lm_head_module="lm_head")
    # generate_interactive(model_new, tok, max_out_len=200, use_logit_lens=True, ln_f_module="model.norm", layer_module_tmp="model.layers.{}", lm_head_module="lm_head")
    
if __name__ == "__main__":
    main()