from ast import literal_eval
import functools
import json
import os
import random
#import wget

# Scienfitic packages
import numpy as np
import nethook
import pandas as pd
import torch
import pdb
from tqdm import tqdm
torch.set_grad_enabled(False)
tqdm.pandas()

# Visuals
from matplotlib import pyplot as plt

# Utilities
from utils import (
    ModelAndTokenizer,
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_from_input,
)
import scipy.stats as stats
plt.style.use('seaborn-whitegrid')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def _split_heads(tensor, num_heads, attn_head_size):
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(1, 0, 2)  # (head, seq_length, head_features)

def _merge_heads(tensor, model):
    num_heads = model.config.n_head
    attn_head_size = model.config.n_embd // model.config.n_head
    
    tensor = tensor.permute(1, 0, 2).contiguous()
    new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
    return tensor.view(new_shape)


def set_act_get_hooks(model, tok_index, attn=False, attn_out=False, mlp=False, mlp_coef=False):
    """
    Only works on GPT2
    """
    # Make sure that these are not set to True at the same time 
    #  so we don't put two different hooks on the same module.  
    assert not (attn is True and attn_out is True)
    
    for attr in ["activations_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_activation(name):
        def hook(module, input, output):
            if "attn" in name:
                if "c_attn" in name:
                    # output.shape: batch_size, seq_len, 3 * hidden_dim
                    _, _, attn_value = output[0].split(model.config.n_embd, dim=1)
                    attn_value = _split_heads(attn_value,
                                              model.config.n_head, 
                                              model.config.n_embd // model.config.n_head)
                    model.activations_[name] = attn_value.detach()
                elif "attn_weights" in name:
                    assert len(output) == 3
                    attn_weights = output[2]  # (batch_size, num_heads, from_sequence_length, to_sequence_length)
                    # the last dimension is a distribution obtained from softmax
                    model.activations_[name] = attn_weights[0][:, tok_index, :].detach()
                else:
                    model.activations_[name] = output[0][:, tok_index].detach()
            elif "m_coef" in name:
                # num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                model.activations_[name] = input[0][:, tok_index].detach()
            elif "m_out" in name:
                model.activations_[name] = output[0][tok_index].detach()
        
        return hook

    hooks = []
    for i in range(model.config.n_layer):
        if attn is True:
            hooks.append(model.transformer.h[i].attn.c_attn.register_forward_hook(get_activation(f"c_attn_value_{i}")))
            hooks.append(model.transformer.h[i].attn.register_forward_hook(get_activation(f"attn_weights_{i}")))
        if attn_out is True:
            hooks.append(model.transformer.h[i].attn.register_forward_hook(get_activation(f"attn_out_{i}")))
        if mlp_coef is True:
            hooks.append(model.transformer.h[i].mlp.c_proj.register_forward_hook(get_activation("m_coef_" + str(i))))
        if mlp is True:
            hooks.append(model.transformer.h[i].mlp.register_forward_hook(get_activation("m_out_" + str(i))))
            
    return hooks


# To block attention edges, we zero-out entries in the attention mask.
# To do this, we add a wrapper around the attention module, because 
# the mask is passed as an additional argument, which could not be fetched 
# with standard hooks before pytorch 2.0.  
def set_block_attn_hooks(model, from_to_index_per_layer, opposite=False):
    """
    Only works on GPT2
    """
    def wrap_attn_forward(forward_fn, model_, from_to_index_, opposite_):
        @functools.wraps(forward_fn)
        def wrapper_fn(*args, **kwargs):
            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for (k, v) in kwargs.items():
                new_kwargs[k] = v

            # hs = args[0]
            # GPT-J
            hs = kwargs["hidden_states"]
            num_tokens = list(hs[0].size())[0]
            num_heads = model_.config.num_attention_heads
            
            if opposite_:
                attn_mask = torch.tril(torch.zeros((num_tokens, num_tokens), dtype=torch.uint8))
                for s, t in from_to_index_:
                    attn_mask[s, t] = 1
            else:
                attn_mask = torch.tril(torch.ones((num_tokens, num_tokens), dtype=torch.uint8))
                for s, t in from_to_index_:
                    attn_mask[s, t] = 0
            attn_mask = attn_mask.repeat(1, num_heads, 1, 1)
            
            attn_mask = attn_mask.to(dtype=model_.dtype)  # fp16 compatibility
            attn_mask = (1.0 - attn_mask) * torch.finfo(model_.dtype).min
            attn_mask = attn_mask.to(hs.device)

            new_kwargs["attention_mask"] = attn_mask
            
            return forward_fn(*new_args, **new_kwargs)

        return wrapper_fn
    
    hooks = []
    for i in from_to_index_per_layer.keys():
        hook = model.transformer.h[i].attn.forward
        model.transformer.h[i].attn.forward = wrap_attn_forward(model.transformer.h[i].attn.forward,
                                                                model, from_to_index_per_layer[i], opposite)
        hooks.append((i, hook))
    
    return hooks


def set_get_attn_proj_hooks(model, tok_index):
    """
    Only works on GPT2
    """
    for attr in ["projs_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_projection(name, E):
        def hook(module, input, output):
            attn_out = output[0][:, tok_index]
            probs, preds = torch.max(
                torch.softmax(attn_out.matmul(E.T), dim=-1), 
                dim=-1
            )
            model.projs_[f"{name}_probs"] = probs.cpu().numpy()
            model.projs_[f"{name}_preds"] = preds.cpu().numpy()
            
        return hook

    E = model.get_input_embeddings().weight.detach()
    hooks = []
    for i in range(model.config.n_layer):
        hooks.append(model.transformer.h[i].attn.register_forward_hook(get_projection(f"attn_proj_{i}", E)))
            
    return hooks


def set_block_mlp_hooks(model, values_per_layer, coef_value=0):
    
    def change_values(values, coef_val):
        def hook(module, input, output):
            output[:, :, values] = coef_val

        return hook

    hooks = []
    for layer in range(model.config.n_layer):
        if layer in values_per_layer:
            values = values_per_layer[layer]
        else:
            values = []
        # hooks.append(model.transformer.h[layer].mlp.c_fc.register_forward_hook(
        #     change_values(values, coef_value)
        # ))
        # GPT-J
        hooks.append(model.transformer.h[layer].mlp.fc_in.register_forward_hook(
            change_values(values, coef_value)
        ))

    return hooks


def set_proj_hooks(model):
    for attr in ["projs_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_projection(name, E):
        def hook(module, input, output):
            num_tokens = list(input[0].size())[1]  #(batch, sequence, hidden_state)
            if name == f"layer_residual_{final_layer}":
                hs = output
            else:
                hs = input[0]
            probs, preds = torch.max(
                torch.softmax(hs.matmul(E.T), dim=-1), 
                dim=-1
            )
            model.projs_[f"{name}_preds"] = preds.cpu().numpy()
            model.projs_[f"{name}_probs"] = probs.cpu().numpy()
        return hook

    E = model.get_input_embeddings().weight.detach()
    final_layer = model.config.n_layer-1
    
    hooks = []
    for i in range(model.config.n_layer-1):
        hooks.append(model.transformer.h[i].register_forward_hook(
            get_projection(f"layer_residual_{i}", E)
        ))
    hooks.append(model.transformer.ln_f.register_forward_hook(
        get_projection(f"layer_residual_{final_layer}", E)
    ))

    return hooks


def set_hs_patch_hooks(model, hs_patch_config, patch_input=False):
    
    def patch_hs(name, position_hs, patch_input):
        
        def pre_hook(module, input):
            for position_, hs_ in position_hs:
                # (batch, sequence, hidden_state)
                input[0][0, position_] = hs_
        
        def post_hook(module, input, output):
            for position_, hs_ in position_hs:
                # (batch, sequence, hidden_state)
                output[0][0, position_] = hs_
        
        if patch_input:
            return pre_hook
        else:
            return post_hook

    hooks = []
    for i in hs_patch_config:
        if patch_input:
            hooks.append(model.transformer.h[i].register_forward_pre_hook(
                patch_hs(f"patch_hs_{i}", hs_patch_config[i], patch_input)
            ))
        else:
            hooks.append(model.transformer.h[i].register_forward_hook(
                patch_hs(f"patch_hs_{i}", hs_patch_config[i], patch_input)
            ))

    return hooks
    

# Always remove your hooks, otherwise things will get messy.
def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()

def remove_wrapper(model, hooks):
    for i, hook in hooks:
        model.transformer.h[i].attn.forward = hook

def cache_subject(mt, knowns_df):
    # create a cache of subject representations

    layers_to_cache = list(range(mt.num_layers+1))
    hs_cache = {}
    for row_i, row in tqdm(knowns_df.iterrows()):
        prompt = row.prompt
        if row.known_id >= 1000:
            break

        inp = make_inputs(mt.tokenizer, [prompt])
        output = mt.model(**inp, output_hidden_states = True)

        for layer in layers_to_cache:
            if (prompt, layer) not in hs_cache:
                hs_cache[(prompt, layer)] = []
            hs_cache[(prompt, layer)].append(output["hidden_states"][layer][0])

    return hs_cache

def attribute_extraction(mt, model_name, knowns_df, loc, known_id=0):
    '''
    Project the hidden states of each layer at the last-relation position to the vocabulary space
    '''
    E = mt.model.get_input_embeddings().weight
    if hasattr(mt.model, "model"):
        lm_head, ln_f = (
            nethook.get_module(mt.model, "lm_head"),
            nethook.get_module(mt.model, "model.norm"),
        )
    else:
        lm_head, ln_f = (
            nethook.get_module(mt.model, "lm_head"),
            nethook.get_module(mt.model, "transformer.ln_f"),
        )
    k = -1
    
    save_path = f"records_{model_name}.json"
    if os.path.exists(save_path):
        with open(save_path,"r") as f0:
            records = json.load(f0)
        return records, mt.num_layers

    hs_cache = cache_subject(mt, knowns_df)
    # Projection of token representations
    print("#hidden states cached#")
    records = []
    for row_i, row in tqdm(knowns_df.iterrows()):
        if known_id and row.known_id != known_id:
            continue
        prompt = row.prompt
        subject = row.subject
        
        inp = make_inputs(mt.tokenizer, [prompt])
        
        record = []
        for layer in range(mt.num_layers):
            # For the detection on the last-relation token
            positions = [(len(inp["input_ids"][0])-1, f"no_subj_last_{layer+1}")]
            for (position, desc) in positions:
                if desc.rsplit("_", 1)[0] != loc:
                    continue
                hs = hs_cache[(prompt, layer)][0][position]
                projs = torch.softmax(
                    lm_head(ln_f(hs)), dim=0
                ).cpu().numpy()
                # projs = hs.matmul(E.T).cpu().numpy()
                ind = np.argsort(-projs)

                if desc.rsplit("_", 1)[0]== loc:
                    record.append({
                        "example_index": row_i,
                        "subject": subject,
                        "layer": layer,
                        "position": position,
                        "desc": desc,
                        "desc_short": desc.rsplit("_", 1)[0],
                        "top_k_preds": [decode_tokens(mt.tokenizer, [i])[0] for i in ind[:k]],
                        "attribute": row.attribute,
                    })
        records.append(record)
        if known_id and row.known_id == known_id:
            break

    tmp = records
    #tmp_s = pd.DataFrame.from_records(record)
    with open(save_path,"w") as f:
        json.dump(records,f,indent=4)
    return tmp, mt.num_layers

def plot_rank_change(record, start_layer, target, t, prompt, scale, location, layer):
    '''
      Plot the rank change across layers
    '''
    st = start_layer
    x = list(range(st,len(record)))
    y = []
    #font
    plt.rcParams['font.family']='Times New Roman, SimSun'
    for r in record:
        if r['layer'] < st:
            continue
        p = r['top_k_preds']
        if target not in p:
            y.append(50000)
        else:
            y.append(p.index(target))
    if len(x)==len(y):
        plt.plot(x, y)
        plt.title(f"{prompt} [{target.strip(' ')}]")
        plt.xlabel('layer')
        plt.ylabel('rank')
        plt.savefig(f"RANK/gpt2_xl_k{scale}_{layer}_{location}_{target.strip(' ')}_{t}_{st}.jpg")
        plt.close()
    else:
        print("Wrong Length!")
    return y

def draw_line(name_of_alg,color_index,datas,x_l):
    palette = plt.get_cmap('Set1')
    color=palette(color_index)
    
    avg=np.mean(datas,axis=0)
    std=np.std(datas,axis=0)
    r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))
    r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))
    plt.plot(x_l, avg, color=color,label=name_of_alg,linewidth=3.5, fontsize=12)
    plt.fill_between(x_l, r1, r2, color=color, alpha=0.2)

def draw_bar(datas, x_l):
    avg2=np.mean(datas,axis=0)
    avg_max = np.max(avg2)
    print(avg_max)
    # std_err=np.std(datas,axis=0)
    # error_params=dict(elinewidth=4,ecolor='coral',capsize=2)
    plt.bar(x_l,avg2,color='orange', fontsize=12)

def plot_rank_change_avg(tmp, num_layers, save_path1, save_path2):
    '''
    plot the average rank flow of the target object tokens and random tokens across layers
    '''
    x = list(range(num_layers))
    plt.rcParams['font.family']='Times New Roman, SimSun'
    
    if os.path.exists(save_path1) and os.path.exists(save_path2):
        all_ranks1 = np.load(save_path1)
        all_ranks2 = np.load(save_path2)
    else:
        all_ranks1 = []
        all_ranks2 = []
        for tmp_s in tmp:
            s_rank1 = []
            s_rank2 = []
            for r in tmp_s:
                target = " " + r['attribute']
                p = r['top_k_preds']
                l = r['layer']
                if l == 0:
                    random_idx = random.randint(0,50000)
                    random_tok = p[random_idx]
                if target not in p:
                    s_rank1.append(len(p)-1)
                else:
                    s_rank1.append(p.index(target))
                s_rank2.append(p.index(random_tok))
            all_ranks1.append(s_rank1)
            all_ranks2.append(s_rank2)
        
        all_ranks1 = np.array(all_ranks1)
        all_ranks2 = np.array(all_ranks2)
        # save array
        np.save(save_path1, all_ranks1)
        np.save(save_path2, all_ranks2)
    
    draw_line("target object",1, all_ranks1, x)
    draw_line("random token",2, all_ranks2, x)
    
    # plt.title("Average Ranking of 1000 Prompts Across Layers for Llama")
    plt.xlabel('layer', fontsize=12)
    plt.ylabel('Token Ranking(-th)', fontsize=12)
    plt.legend(frameon=True)
    plt.savefig(f"avg_rank_flow_1000.pdf")
    plt.close()

    return

def attributes_rate(tmp, num_layers, mt, save_path):
    '''
      plot the average attributes rate bars at the last-relation position across layers 
    '''
    x = list(range(num_layers))
    plt.rcParams['font.family']='Times New Roman, SimSun'
    
    k = 50
    if os.path.exists(save_path):
        a_r = np.load(save_path)
    else:
        with open("relation_related_attributes.json","r") as f:
            r_attribtues = json.load(f)
        with open("known_1000.json","r") as f1:
            known_df = json.load(f1)
        relation_ids = r_attribtues.keys()
        processed_attributes = {}
        for r_l in relation_ids:
            attributes = r_attribtues[r_l]
            tokenzied_attibutes = []
            for a in attributes:
                if hasattr(mt.model, "model"):
                    a_ids = mt.tokenizer(f"{a}")["input_ids"][1:]
                else:
                    a_ids = mt.tokenizer(f" {a}")["input_ids"]
                for a_id in a_ids:
                    tokenzied_a = mt.tokenizer.decode(a_id)
                    tokenzied_attibutes.append(tokenzied_a)
            processed_attributes[r_l] = tokenzied_attibutes
        
        a_r = []
        for tmp_s_idx in range(len(tmp)):
            tmp_s = tmp[tmp_s_idx]
            a_r_s = []
            relation_id = known_df[tmp_s_idx]["relation_id"]
            if relation_id not in relation_ids:
                continue
            for r in tmp_s:
                p = r['top_k_preds'][:k]
                c = 0
                for t in p:
                    if t in processed_attributes[relation_id]:
                        c += 1
                a_r_s.append(c)    
            a_r.append(a_r_s) 
        
        a_r = np.array(a_r) / k
        np.save(save_path ,a_r)
    
    a_r = a_r * 100
    draw_bar(a_r, x)
    
    plt.xlabel('layer', fontsize=12)
    plt.ylabel('Attributes Rate(%)', fontsize=12)
    plt.savefig(f"avg_attributes_rate_1000.pdf")
    plt.close()
    return

def spearman_compute():
    '''
    compute the spearman coefficient between the avg rank change of the target tokens and the avg attributes rates
    '''
    all_ranks1 = np.load("results/rank_tar_gpt-j-6b.npy")
    a_r = np.load("results/ar_gptj_lr.npy")
    
    avg_all_ranks1 = -np.mean(all_ranks1,axis=0)
    avg_a_r = np.mean(a_r,axis=0)
    spearman = stats.spearmanr(avg_all_ranks1, avg_a_r)
    print(f"spearman coefficient: {spearman}") 

def sublayer_knockout(mt, knowns_df, model_name):
    '''
    Projection of token representations while applying knockouts to MHSA/MLP sublayers
    '''
    def cal_attributes_rate(top_k_pred, attributes):
        c = 0.0
        for t in top_k_pred:
            if t in attributes:
                c += 1 
        return c/len(top_k_pred)  
    k = 50
    E = mt.model.get_input_embeddings().weight
    all_mlp_dims = list(range(mt.model.config.n_embd * 4))
    relation_repr_layer = 24
    num_block_layers = 10
    sl = 100
    with open(f"processed_related_attributes_{model_name}.json","r") as f:
        r_attribtues = json.load(f)

    b_attn_ars = []
    b_mlp_ars = []
    c = 0
    for row_i, row in tqdm(knowns_df.iterrows()):
        if row.relation_id not in r_attribtues.keys():
            continue
        c += 1
        if c > 100:
            break
        prompt = row.prompt
        subject = row.subject
        relation_id = row.relation_id
        attributes = r_attribtues[relation_id]
        
        inp = make_inputs(mt.tokenizer, [prompt])
        e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], prompt)
        e_range = [x for x in range(e_range[0], e_range[1])]
        position = e_range[-1]
        
        output_ = mt.model(**inp, output_hidden_states = True)
        hs_ = output_["hidden_states"][relation_repr_layer+1][0, position]
        projs_ = hs_.matmul(E.T).cpu().numpy()
        ind_ = np.argsort(-projs_)
        top_k_preds_ = [decode_tokens(mt.tokenizer, [i])[0] for i in ind_[:k]]
        
        b_attn_ar = []
        b_mlp_ar = []
        for start_block_layer in range(relation_repr_layer):
            ## attributes rate
            raw_ar = cal_attributes_rate(top_k_preds_, attributes)
            
            end_block_layer = min(start_block_layer + num_block_layers + 1, relation_repr_layer)
            block_layers = [l for l in range(start_block_layer, end_block_layer)]
            for block_module in ["mlp", "attn"]:
                with torch.no_grad():
                    if block_module == "mlp":
                        block_config = {layer_: all_mlp_dims for layer_ in block_layers}
                        block_mlp_hooks = set_block_mlp_hooks(mt.model, block_config)
                        output = mt.model(**inp, output_hidden_states = True)
                        remove_hooks(block_mlp_hooks)
                    elif block_module == "attn":
                        block_config = {layer_: [] for layer_ in block_layers}
                        block_attn_hooks = set_block_attn_hooks(mt.model, block_config, opposite=True)
                        output = mt.model(**inp, output_hidden_states = True)
                        remove_wrapper(mt.model, block_attn_hooks)

                hs = output["hidden_states"][relation_repr_layer+1][0, position]
                projs = hs.matmul(E.T).cpu().numpy()
                ind = np.argsort(-projs)
                top_k_preds_kind = [decode_tokens(mt.tokenizer, [i])[0] for i in ind[:k]]
                if block_module == "mlp":
                    mlp_ar = cal_attributes_rate(top_k_preds_kind, attributes)
                    b_mlp_ar.append(raw_ar - mlp_ar)
                elif block_module == "attn":
                    attn_ar = cal_attributes_rate(top_k_preds_kind, attributes)
                    b_attn_ar.append(raw_ar - attn_ar)
        b_mlp_ars.append(b_mlp_ar)
        b_attn_ars.append(b_attn_ar)
        
    b_mlp_ars = np.array(b_mlp_ars).mean(axis=0)
    b_attn_ars = np.array(b_attn_ars).mean(axis=0)
    np.save("b_mlp_ars_j.npy",b_mlp_ars)
    np.save("b_attn_ars_j.npy",b_attn_ars)
    
    return b_mlp_ars, b_attn_ars

def preprocess_attributes(mt, model_name):
    '''
      Tokenize the ground-truth relation related attributes for gpt2-xl/gpt-j-6b
    '''
    with open("relation_related_attributes.json","r") as f:
        r_attribtues = json.load(f)
    relation_ids = r_attribtues.keys()    
        
    processed_attributes = {}
    for r_l in relation_ids:
        attributes = r_attribtues[r_l]
        tokenzied_attibutes = []
        for a in attributes:
            a_ids = mt.tokenizer(f" {a}")["input_ids"]
            for a_id in a_ids:
                tokenzied_a = mt.tokenizer.decode(a_id)
                tokenzied_attibutes.append(tokenzied_a)
        processed_attributes[r_l] = tokenzied_attibutes
    with open(f"processed_related_attributes_{model_name}.json","w") as f2:
        json.dump(processed_attributes,f2,indent=2)

def plot_blocked_AR(b_mlp_ars, b_attn_ars, num_layers):
    '''
      Plot the avg attributes rate after knockouts of mlp/attn sublayers 
    '''
    plt.rcParams['font.family']='Times New Roman, SimSun'
    x = list(range(0,num_layers))
    y_raw = [0] * num_layers
    y_b_mlp = list(-b_mlp_ars) + [0.0,0.0,0.0]
    y_b_attn = list(-b_attn_ars) + [0.0,0.0,0.0]
    plt.plot(x, y_raw, 'b', label="original")
    plt.plot(x, y_b_mlp, 'g', label ="blocking MLP")
    plt.plot(x, y_b_attn, 'r', label ="blocking ATTN")
    plt.xlabel('layer', fontsize=16)
    plt.ylabel('AR decline', fontsize=14)
    plt.legend(fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.savefig("knockout_ar.pdf")
    plt.close()
    return

def main():
    # Setup
    loc = "no_subj_last"
    knowns_df = pd.read_json("known_1000.json")
    results_dir = "results/"
    
    ## Load model
    model_name = "gpt2-xl" # "llama-2-7b" or "gpt-j-6b"
    mt = ModelAndTokenizer(
        model_name,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    )
    mt.model.eval()
    
    print("---raw attributes extraction---")
    tmp, num_layers = attribute_extraction(mt, model_name, knowns_df, loc, 0)
    
    print("---plot avg rank flow---")
    target_rank_path = results_dir + f"rank_tar_{model_name}.npy"
    random_rank_path = results_dir + f"rank_rand_{model_name}.npy"
    plot_rank_change_avg(tmp, num_layers, target_rank_path, random_rank_path)
    
    print("---plot avg AR---")
    attributes_rate_path = results_dir + f"ar_{model_name}.npy"
    attributes_rate(tmp, num_layers, mt, attributes_rate_path)
        
    print("---attributes rate with mlp/attn module knocked out---")
    block_attn_path = results_dir + f"b_attn_{model_name}.npy"
    block_mlp_path = results_dir + f"b_mlp_{model_name}.npy"
    if os.path.exists(block_attn_path) and os.path.exists(block_mlp_path):
        b_mlp_ars = np.load(block_mlp_path)
        b_attn_ars = np.load(block_attn_path)
    else:
        preprocess_attributes(mt, model_name)
        b_mlp_ars, b_attn_ars =sublayer_knockout(mt, knowns_df, model_name)
    plot_blocked_AR(b_mlp_ars,b_attn_ars,mt.num_layers)
    
    
if __name__ =="__main__":
    main()
    # spearman_compute()