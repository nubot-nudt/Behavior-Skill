"""Load a policy from a local hydra config."""

import os
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf



def load_hydra_policy(policy_cfg_path: str):
    """Instantiate a policy (policy_cfg_loaded.model) from a hydra config directory."""
    with hydra.initialize_config_dir(policy_cfg_path, version_base="1.1"):
        policy_cfg_loaded = hydra.compose(os.path.basename(policy_cfg_path))
    OmegaConf.resolve(policy_cfg_loaded)
    return instantiate(policy_cfg_loaded.model)
