import os
import pdb
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rets import RETSHyperParams, apply_rets_to_model
from util import nethook
from util.generate import generate_fast
from util.globals import *


def demo_model_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    generation_prompts: List[str],
    alg_name: str = "RETS",
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Applies the selected model editing algorithm. Generates text both before and after
    for comparison of model behavior. Returns the updated model and the original values of
    weights that were changed.
    """

    nethook.set_requires_grad(True, model)

    RewritingParamsClass, apply_method, hparams_prefix, hparams_suffix = load_alg(
        alg_name
    )
    params_name = (
        HPARAMS_DIR
        / hparams_prefix
        / f"{model.config._name_or_path.replace('/', '_')}{hparams_suffix}.json"
    )

    print_loud("Generating pre-update text")
    pre_update_text = generate_fast(model, tok, generation_prompts, max_out_len=100)
    print(pre_update_text)

    print_loud(f"Retrieving {alg_name} hyperparameters")
    print("Loading from", params_name)
    hparams = RewritingParamsClass.from_json(params_name)
    print(hparams)
    
    print_loud(f"Applying {alg_name} to model")
    model_new, orig_weights = apply_method(
        model, tok, requests, hparams, return_orig_weights=False
    )

    print_loud("Generating post-update text")
    post_update_text = generate_fast(
        model_new, tok, generation_prompts, max_out_len=100
    )
    print(post_update_text)

    print_loud("Summarizing differences")
    for i, (prompt, pre, post) in enumerate(
        zip(generation_prompts, pre_update_text, post_update_text)
    ):
        if i > 0:
            print("".join(["-" for _ in range(10)]))

        prompt_str = "[Prompt]:"
        pre_str = f"[Pre-{alg_name}]:"
        post_str = f"[Post-{alg_name}]:"
        pad_to = 1 + max(len(prompt_str), len(pre_str), len(post_str))

        for s, t in zip([prompt_str, post_str, pre_str], [prompt, post, pre]):
            print(s.ljust(pad_to), t)

    return model_new, orig_weights

def load_alg(alg_name):
    """
    Loads dependencies for the desired algorithm.
    Implementation is slightly awkward to prevent unnecessary imports on Colab.

    The return value is a tuple of the following:
    1. Class for storing hyperparameters
    2. Method for applying rewrites
    3. Location of parameters
    4. Predefined suffix for the param file
    """
    assert alg_name in [
        "ROME",
        "RETS"
    ]

    if alg_name == "RETS":
        return RETSHyperParams, apply_rets_to_model, "RETS", ""


def print_loud(x, pad=3):
    """
    Prints a string with # box for emphasis.

    Example:
    ############################
    #                          #
    #  Applying ROME to model  #
    #                          #
    ############################
    """

    n = len(x)
    print()
    print("".join(["#" for _ in range(n + 2 * pad)]))
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print(
        "#"
        + "".join([" " for _ in range(pad - 1)])
        + x
        + "".join([" " for _ in range(pad - 1)])
        + "#"
    )
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print("".join(["#" for _ in range(n + 2 * pad)]))


class StopExecution(Exception):
    def _render_traceback_(self):
        pass


def stop_execution():
    raise StopExecution
