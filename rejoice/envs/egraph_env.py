import numpy as np

import gym
from gym import spaces
from ..rejoice import *
from collections import OrderedDict, deque, namedtuple
from ..lib import Language
from typing import Tuple, Optional, Union
from ..graph_space import GraphSpace
import torch
import math


class EGraphEnv(gym.Env):
    """Custom gym env for the egraph rule selection task."""
    metadata = {'render.modes': []}

    def __init__(self, lang: Language, expr: any):
        super(EGraphEnv, self).__init__()
        self.lang = lang
        self.expr = expr
        self.rewrite_rules = lang.rewrite_rules()
        self.action_space = spaces.Discrete(len(self.rewrite_rules) + 1,)
        self.observation_space = GraphSpace(num_node_features=lang.num_node_features,
                                            low=0,
                                            high=lang.get_feature_upper_bounds())
        self.reward_range = (-1, 1)
        self.egraph, self.max_cost, self.prev_cost = None, None, None

    def step(self, action: any) -> Tuple[any, float, bool, dict]:
        info = {"actual_cost": self.prev_cost}
        is_stop_action = action == len(self.rewrite_rules)
        if is_stop_action:
            # Agent has chosen to stop optimizing and terminate current episode
            # punish agent if it ends the episode without any improvement
            reward = -1.0 if self.prev_cost == self.max_cost else 0.0
            # TODO: should the next obs be None or still the current state?
            # return self._get_obs(), reward, True, info
            return None, reward, True, info

        rewrite_to_apply = [self.rewrite_rules[action]]
        stop_reason = self.egraph.run(rewrite_to_apply, iter_limit=1, node_limit=10000)
        info["stop_reason"] = stop_reason
        if stop_reason == 'SATURATED':
            # if it was saturated, applying the rule did nothing; no need to re-extract
            reward = -0.5
        elif stop_reason == 'NODE_LIMIT':
            reward = -1.0
        else:
            best_cost, best_expr = self.egraph.extract(self.expr)
            best_cost = float(best_cost)
            info["actual_cost"] = best_cost
            if best_cost == self.prev_cost:
                # expanded egraph, but didn't get us a better extraction cost
                reward = -0.05
            else:
                reward = (self.prev_cost - best_cost) / self.max_cost
                self.prev_cost = best_cost
            if stop_reason != "ITERATION_LIMIT":
                reward -= 1.0  # punish for timing out

        is_done = is_terminal(stop_reason)
        if stop_reason != "NODE_LIMIT":
            new_obs = self._get_obs()
        else:
            new_obs = None

        return new_obs, reward, is_done, info

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[any, tuple[any, dict]]:
        self.egraph = EGraph()
        self.egraph.add(self.expr)
        self.max_cost = float(self.egraph.extract(self.expr)[0])
        self.prev_cost = self.max_cost
        # reward is normalized to (0, max_cost)
        new_obs = self._get_obs()
        return new_obs

    def _get_obs(self):
        return self.lang.encode_egraph(self.egraph)

    def close(self):
        pass


def is_terminal(stop_reason: str):
    """The episode should end if egg returns a STOP_REASON that indicates that the egraph has grown
        too large or extraaction is timing out."""
    if stop_reason == "ITERATION_LIMIT":
        # This means that the rewrite we took succeeded and nothing unexpected happened.
        return False
    elif stop_reason == "SATURATED":
        # Note that SATURATION isn't global saturation; it just means that the action we took didn't
        # change the egraph. This will happen frequently and is normal.
        return False
    else:
        return True

