from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class RETSHyperParams(HyperParams):
    # Method
    layers: List[int]
    device: str
    stats_dir: str
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    context_template_length_params: List[List[int]]
    nll_loss_factor: float
    origin_mom: bool
    subject_const: bool

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
