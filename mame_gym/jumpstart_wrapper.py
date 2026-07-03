"""jumpstart_wrapper.py — JSRL (Jump-Start RL, Uchendu et al. 2022).

The FSM guide drives the first `h` steps of each episode; the RL policy takes over for the
rest. `h` anneals per-episode from h0 -> 0, so the learner first practices the DEEP states
the FSM reaches, then progressively earlier ones — an automatic curriculum that also lets
the learner EXCEED the guide (unlike imitation, which is capped at the guide). Cures the
covariate-shift collapse because the learner trains on-policy from real reachable states.

Wraps the raw MameRobotronEnv (below VecNormalize) so the FSM sees the raw 945-dim obs.
"""
import gymnasium as gym
import numpy as np
from obs_decode import fsm_action_from_obs


class JumpStartWrapper(gym.Wrapper):
    def __init__(self, env, h0=500, anneal_eps=1200, h_min=0):
        super().__init__(env)
        self.h = float(h0)
        self.dec = h0 / max(1, anneal_eps)
        self.h_min = h_min

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        h = int(self.h)
        for _ in range(h):
            mv, fr = fsm_action_from_obs(obs)
            obs, _, term, trunc, info = self.env.step(np.array([mv, fr], dtype=np.int64))
            if term or trunc:            # FSM died during the guided prefix → fresh episode
                obs, info = self.env.reset(**kw)
                break
        self.h = max(self.h_min, self.h - self.dec)
        info = dict(info); info["jsrl_h"] = int(self.h)
        return obs, info

    def step(self, action):
        return self.env.step(action)
