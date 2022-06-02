from .PretrainingDataset import PretrainingDataset
import os
import time
import torch
from .rejoice import EGraph
from .lib import Language
import numpy as np
import sys
from pathlib import Path
import itertools


class EGraphSolver:

    def __init__(self, lang: Language, expr: any, node_limit=10_000, rng=np.random.default_rng()):
        self.lang = lang
        self.expr = expr
        self.rewrite_rules = lang.rewrite_rules()
        self.node_limit = node_limit
        self.rng = rng

    def optimize(self, max_steps=500) -> "list[int]":
        best_possible_cost, best_possible_expr = self.exhaustive_search()
        # print("egg cost:", best_possible_cost, "expr", best_possible_expr)

        # try to re-create egg's cost, this time tracking the actions at each step
        egraph = self.new_egraph()
        steps = []
        for i in range(max_steps):
            action, stop_reason, cost = self.step(
                egraph, self.rng.integers(0, len(self.rewrite_rules)))
            if stop_reason == 'NODE_LIMIT':
                print("hit node limit when searching...")
                raise Exception
            elif stop_reason != 'SATURATED':  # if 'SATURATED', the step didn't change the egraph; we can filter it
                steps.append(action)
                if cost <= best_possible_cost:
                    break  # we don't need to take any more steps; we found the min cost

        # now extract the minimum action order needed to get this cost
        all_poss_seqs = list(itertools.product([0, 1], repeat=len(steps)))
        # sorting by sum means that we try the smallest action sequence lengths first
        all_poss_seqs.sort(key=sum)
        egraph_base = self.new_egraph()
        for seq_mask in all_poss_seqs:
            eg = egraph_base.clone()
            actions = list(itertools.compress(steps, seq_mask))  # mask
            for action in actions:
                stop_reason = eg.run(
                    [self.rewrite_rules[action]], iter_limit=1, node_limit=self.node_limit)
                if stop_reason != 'NODE_LIMIT':
                    cost, ex = eg.extract(self.expr)
                    if cost == best_possible_cost:  # found the shortest action sequence to achieve cost equiv to egg
                        self.build_pyg_data(actions)
                        return actions

    def build_pyg_data(self, actions: "list[int]"):
        """Convert an action sequence to a list of PyTorch Geometric data objects."""
        egraph = self.new_egraph()
        for action in actions:
            data = self.lang.encode_egraph(egraph, action)
            lang_name = self.lang.name
            if not os.path.exists(lang_name):
                os.makedirs(lang_name)
            torch.save(
                data, f'{lang_name}/{lang_name}_a{action}_n{len(data.x)}_e{len(data.edge_index[0])}_{time.strftime("%Y%m%d-%H%M%S")}.pt')
            egraph.run([self.lang.rewrite_rules()[action]],
                       iter_limit=1, node_limit=10_000)

        # Add termination action
        end_action_ind = self.lang.num_rules
        data = self.lang.encode_egraph(egraph, end_action_ind)
        torch.save(
            data, f'{lang_name}/{lang_name}_a{action}_n{len(data.x)}_e{len(data.edge_index[0])}_{time.strftime("%Y%m%d-%H%M%S")}.pt')

    def step(self, egraph: EGraph, action: int):
        rewrite_to_apply = [self.rewrite_rules[action]]
        stop_reason = egraph.run(
            rewrite_to_apply, iter_limit=1, node_limit=self.node_limit)
        best_cost, best_expr = egraph.extract(self.expr)
        best_cost = float(best_cost)
        return action, stop_reason, best_cost

    def exhaustive_search(self, iter_limit=7):
        egraph = self.new_egraph()
        egraph.run(self.lang.rewrite_rules(), iter_limit=iter_limit,
                   node_limit=self.node_limit, use_backoff=True)
        best_cost, best_expr = egraph.extract(self.expr)
        return best_cost, best_expr

    def new_egraph(self):
        egraph = EGraph()
        egraph.add(self.expr)
        return egraph


def generate_dataset(lang: Language, num=10, rng=np.random.default_rng()):
    exprs = [lang.gen_expr(p_leaf=0.1) for i in range(num)]

    for ind, expr in enumerate(exprs):
        print("Generating expr", ind, expr)
        solver = EGraphSolver(lang, expr)
        try:
            solver.optimize()
        except Exception as e:
            print("Failed to solve expr", ind, expr)
            print(e)
            continue
