import json
import os
import shutil
from pathlib import Path
from time import time
from typing import Tuple, Union
import numpy as np
import pdb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from baselines.efk import EFKHyperParams, EfkRewriteExecutor
from .dsets import (
    AttributeSnippets,
    CounterFactDataset,
    get_tfidf_vectorizer,
)
from .py.eval_utils_counterfact import compute_rewrite_quality_counterfact_rs
from rome import ROMEHyperParams, apply_rome_to_model
from rets import RETSHyperParams, apply_rets_to_model
from util import nethook
from util.globals import *

ALG_DICT = {
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "RETS": (RETSHyperParams, apply_rets_to_model)
}

DS_DICT = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact_rs)
}

CUDA = "cuda:4"

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    conserve_memory: bool,
    dir_name: str,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    if continue_from_run is not None:
        run_dir = RESULTS_DIR / dir_name / continue_from_run
        assert (
            run_dir.exists()
        ), f"If continuing from run, {continue_from_run} must exist!"
    else:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    print("Instantiating model")
    if type(model_name) is str:
        if 'llama' in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(model_name).to(CUDA)
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(CUDA)
            tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        if 'llama' in model_name.lower():
            tok.padding_side = 'right'
    else:
        model, tok = model_name

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, size=dataset_size_limit, tok=tok)

    # Iterate through dataset
    for record in ds:
        case_id = record["case_id"]
        print("#############")
        print(case_id)
        print("#############")
        case_result_path = run_dir / f"case_{case_id}.json"
        if not case_result_path.exists():
            # Compute weight changes + record weights that changed
            start = time()
            args_conserve_memory = (
                dict(return_orig_weights_device=("cpu" if conserve_memory else CUDA))
                if conserve_memory
                else dict()
            )
            
            # obtain rs initial tokens
            rs_topk_idx, rs_mask, rs_best = rs_probability_topk(model, tok, record["relation_specific_prompts"]) # l * 10
            record["rs_topk"] = rs_topk_idx
            record["rs_mask"] = rs_mask
            record["rs_best"] = rs_best
            
            edited_model, weights_copy = apply_algo(
                model,
                tok,
                [record["requested_rewrite"]],
                hparams,
                copy=False,
                return_orig_weights=True,
                **args_conserve_memory,
            )
            exec_time = time() - start
            print("Execution took", exec_time)
            
            # Execute evaluation suite
            start = time()
            metrics = {
                "case_id": case_id,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
            }
            
            metrics["post"] = ds_eval_method(edited_model, tok, record, snips, vec)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to(CUDA)
            metrics["pre"] = ds_eval_method(model, tok, record, snips, vec)

            print("Evaluation took", time() - start)

            # Dump metrics in .json
            with open(case_result_path, "w") as f:
                json.dump(metrics, f, indent=1)


def rs_probability_topk(model, tok, relation_specific_prompts):
    k = 20
    E = model.get_input_embeddings().weight
    inp = make_inputs(tok, relation_specific_prompts) #l * d
    output_logits = model(**inp).logits.detach().cpu().numpy()
    projs = output_logits[:,-1,:] # l * d
    rs_prob_idx = np.argsort(-projs)[:,:k]
    
    stop_words = []
    with open("data/stopwords.txt", "r") as f:
        for line in f:
            stop_words.append(line.strip("\n").strip(" "))
    
    rs_mask = []
    rs_best = []
    for rs_p_id in rs_prob_idx:
        mask_tmp = [1] * len(rs_p_id)
        for r in range(len(rs_p_id)):
            r_token = tok.decode(rs_p_id[r]).strip(" ")
            if r_token not in stop_words and r_token!="":
                mask_tmp[r] = 0
                rs_best.append(rs_p_id[r])
                break
        if mask_tmp == [1] * len(rs_p_id):
            mask_tmp[0] = 0
            rs_best.append(rs_p_id[0])
        rs_mask.extend(mask_tmp)
    assert len(rs_mask) == len(rs_best) * k
    return rs_prob_idx, rs_mask, rs_best

def make_inputs(tokenizer, prompts, device=CUDA):
  """Prepare inputs to the model."""
  token_lists = [tokenizer.encode(p) for p in prompts]
  maxlen = max(len(t) for t in token_lists)
  if "[PAD]" in tokenizer.all_special_tokens:
    pad_id = tokenizer.all_special_ids[
        tokenizer.all_special_tokens.index("[PAD]")
        ]
  else:
    pad_id = 0
  input_ids = [
      [pad_id] * (maxlen - len(t)) + t for t in token_lists]
  attention_mask = [
      [0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists
      ]
  return dict(
      input_ids=torch.tensor(input_ids).to(device),
      attention_mask=torch.tensor(attention_mask).to(device),
      )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["ROME", "FT", "KN", "MEND", "KE", "MEMIT"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "zsre"],
        default="cf",
        help="Dataset to perform evaluations on. Either CounterFact (cf) or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=10000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.conserve_memory,
        dir_name=args.alg_name,
    )
