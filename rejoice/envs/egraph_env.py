import numpy as np

import gym
from gym import spaces
from ..rejoice import *
from ..lib import Language
from typing import Tuple, Optional, Union
import torch
import math


class EGraphEnv(gym.Env):
    """Custom gym env for the egraph rule selection task."""
    metadata = {'render.modes': []}

    def __init__(self, lang: Language, egraph: EGraph, expr: any):
        super(EGraphEnv, self).__init__()
        self.lang = lang
        self.expr = expr
        self.egraph = egraph
        self.rewrite_rules = lang.rewrite_rules()
        self.num_node_features = lang.num_node_features()
        self.action_space = spaces.Discrete(len(self.rewrite_rules), )

        # self.observation_shape = (lang.num_node_features(),)
        # self.observation_space = spaces.Box(low=0.0, high=1.0, shape=self.observation_shape)

    def step(self, action: any) -> Tuple[any, float, bool, dict]:
        # ask policy network for action
        rewrite_to_apply = [self.rewrite_rules[action[0]]]
        # apply action to egraph
        self.egraph.run(rewrite_to_apply)

        # reward is negative square root of best cost
        best_cost, best_expr = self.egraph.extract(self.expr)
        reward: float = -math.sqrt(best_cost)
        new_obs = self.lang.encode_egraph(self.egraph)
        # TODO: Stop when saturated
        is_done = False

        return new_obs, reward, is_done, dict()

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[any, tuple[any, dict]]:
        # reset egraph
        self.egraph = EGraph(self.lang.eclass_analysis)
        self.egraph.add(self.expr)

        # return the new observation
        new_obs = self.lang.encode_egraph(self.egraph)
        return new_obs



