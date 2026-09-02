"""Remote policy integration over WebSocket."""

import torch as _th
import numpy as _np

class WebsocketPolicyWrapper:
    """Wrap WebsocketClientPolicy.act() behind a forward(obs, instruction) interface."""

    def __init__(self, host, port):
        from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
        self._client = WebsocketClientPolicy(host=host, port=port)

    def forward(self, obs, instruction=None):
        # msgpack's pack_array only supports numpy; torch tensors must be converted first
        obs_packed = {}
        for k, v in obs.items():
            if isinstance(v, _th.Tensor):
                obs_packed[k] = v.cpu().numpy()
            else:
                obs_packed[k] = v
        # inject the instruction as the prompt field (msgpack natively supports strings)
        if instruction is not None:
            obs_packed["prompt"] = instruction
        return self._client.act(obs_packed)

    def reset(self):
        self._client.reset()
