import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import repr_tools
from util.globals import *

from .layer_stats import layer_stats
from .rets_hparams import RETSHyperParams

from util import nethook

# Cache variables
inv_mom2_cache = {}

def get_inv_cov_SC(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    hparams: RETSHyperParams,
    # origin_mom: bool=True,
    # subject_const: bool=True
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    origin_mom : whether to use the original cov matrix from Wikipedia
    subject_const: whether to use subject constraints during editing
    """

    global inv_mom2_cache

    mom2_dataset = hparams.mom2_dataset
    mom2_n_samples = hparams.mom2_n_samples
    mom2_dtype = hparams.mom2_dtype
    layer_name = hparams.rewrite_module_tmp.format(layer)

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    
    if key not in inv_mom2_cache:
        print(
            f"Retrieving inverse covariance statistics for {model_name} @ {layer_name}. "
            f"The result will be cached to avoid repetitive computation."
        )
        if hparams.origin_mom:
            stat = layer_stats(
                model,
                tok,
                layer_name,
                STATS_DIR,
                mom2_dataset,
                to_collect=["mom2"],
                sample_size=mom2_n_samples,
                precision=mom2_dtype,
                download=False,
                hparams=hparams
            )
            # origin mom2
            stat_mom = stat.mom2.moment().to(f"cuda:{hparams.device}")
        if hparams.subject_const:
            # subject constraints
            c = get_rel_cov_from_known(model, tok, layer, context_templates, hparams).to(f"cuda:{hparams.device}")
        
        # get inverse cov   
        if hparams.origin_mom and hparams.subject_const:
            cov = stat_mom + c
        elif hparams.origin_mom:
            cov = stat_mom
        else:
            cov = c

        inv_mom2_cache[key] = torch.inverse(cov).float()  # Cast back to float32

    return inv_mom2_cache[key]


def compute_ur(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: RETSHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    print("Computing left vector (u)...")
    
    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request["subject"]
        print(f"Selected u projection object {word}")
        cur_repr = repr_tools.get_reprs_at_word_tokens(
            context_templates=[
                templ.format(request["prompt"]) for templ in context_templates
            ],
            words=[word for _ in range(len(context_templates))],
            subtoken=hparams.fact_token[len("subject_") :],
            **word_repr_args,
        ).mean(0)
    elif hparams.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    # Apply inverse second moment adjustment
    u = cur_repr
    if hparams.mom2_adjustment:
        u = get_inv_cov_SC(
            model,
            tok,
            layer,
            context_templates,
            hparams
        ) @ u.unsqueeze(1)
        
        u = u.squeeze()

    return u / u.norm()

def get_rel_cov_from_known(model, tok, layer, context_templates, hparams):
    ## compute the matrix for subject constraints
    known_name = "counterfact"
    model_name = model.config._name_or_path.replace("/", "_")
    knowns_df = pd.read_json(f"data/{known_name}.json")
    
    stats_dir = Path(hparams.stats_dir)
    file_path = stats_dir / model_name
    save_path = file_path / f"{known_name}_c_{str(layer)}.pt"
    # save_path = f"{hparams.stats_dir}/{model_name}/{known_name}_c_{str(layer)}.pt"
    
    mlp_module_tmp = hparams.rewrite_module_tmp
    
    if not file_path.exists():
        file_path.mkdir(exist_ok=True, parents=True)
    if os.path.exists(save_path):
        c = torch.load(save_path, weights_only=True)
        return c
    
    context_templates = ["{}"]
    k = None
    for row_i, row in tqdm(knowns_df.iterrows()):
        subject = row.requested_rewrite["subject"]
        prompt = row.requested_rewrite["prompt"].format(subject)
        temp_prompts = [
            templ.format(prompt) for templ in context_templates
        ]
        
        k_prompt = None
        for temp_prompt in temp_prompts:
            inp = make_inputs(tok, [temp_prompt], f"cuda:{hparams.device}")
            position = len(inp["input_ids"][0])-1
            with nethook.Trace(
                model, layer=mlp_module_tmp.format(layer), retain_input=True, retain_output=False, stop=True
            ) as tr:
                model(**inp)
            if k_prompt is not None:
                k_prompt = torch.cat((k_prompt, tr.input[0][position].detach().unsqueeze(0)), 0)
            else:
                k_prompt = tr.input[0][position].detach().unsqueeze(0)
        if k_prompt==None:
            continue
        k_prompt_mean = torch.mean(k_prompt, dim=0)
        if k is not None:
            k = torch.cat((k,k_prompt_mean.unsqueeze(0)), 0)
        else:
            k = k_prompt_mean.unsqueeze(0)
              
    print(f"Subject Constraint Matrix size: {k.size()}")
    c = torch.matmul(k.T, k)
    torch.save(c, save_path)
    return c

def make_inputs(tokenizer, prompts, device="cuda"):
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
  
def find_token_range(tokenizer, token_array, substring):
  """Find the tokens corresponding to the given substring in token_array."""
  toks = decode_tokens(tokenizer, token_array)
  whole_string = "".join(toks)
  char_loc = whole_string.index(substring)
  loc = 0
  tok_start, tok_end = None, None
  for i, t in enumerate(toks):
    loc += len(t)
    if tok_start is None and loc > char_loc:
      tok_start = i
    if tok_end is None and loc >= char_loc + len(substring):
      tok_end = i + 1
      break
  return (tok_start, tok_end)

def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]