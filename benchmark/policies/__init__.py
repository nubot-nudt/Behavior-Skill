"""Policy integration layer.

The evaluator (SkillEvaluator / run_batch) only relies on duck typing:

    policy.reset()                                     # called before each episode
    action = policy.forward(obs=obs, instruction=...)  # per-step inference

Two integration options:
    websocket.WebsocketPolicyWrapper  — connect to a remote policy server (model loaded server-side)
    hydra_policy.load_hydra_policy    — instantiate a policy from a local hydra config
"""

from .hydra_policy import load_hydra_policy
from .websocket import WebsocketPolicyWrapper

__all__ = ["WebsocketPolicyWrapper", "load_hydra_policy"]
