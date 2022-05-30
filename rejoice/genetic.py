from typing import Tuple, Optional, Union
from ortools.linear_solver import pywraplp
from collections import namedtuple
import numpy as np
import sys
from pathlib import Path
import itertools
import pygad
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import Language
from rejoice import EGraph
from tests.test_lang import TestLang

Step = namedtuple('Step', ['action', 'cost'])
SentinelStep = namedtuple('SentinelStep', ['action', 'cost', 'step_ind'])

seed = 1
np.random.seed(seed)


class GeneticSolver:
    """The goal is"""

    def __init__(self, lang: Language, expr: any, node_limit=10_000):
        self.lang = lang
        self.expr = expr
        self.rewrite_rules = lang.rewrite_rules()
        self.node_limit = node_limit

    def prim_optimize(self, max_steps=500) -> list[int]:
        best_possible_cost, best_possible_expr = self.exhaustive_search()
        print("egg cost:", best_possible_cost, "expr", best_possible_expr)

        # try to re-create egg's cost, this time tracking the actions at each step
        egraph = self.new_egraph()
        steps = []
        for i in range(max_steps):
            action, stop_reason, cost = self.step(egraph, np.random.randint(0, len(self.rewrite_rules)))
            if stop_reason != 'SATURATED':  # if 'SATURATED', the step didn't change the egraph; we can filter it
                steps.append(action)
                if cost <= best_possible_cost:
                    break  # we don't need to take any more steps; we found the min cost

        # now extract the minimum action order needed to get this cost
        all_poss_seqs = list(itertools.product([0, 1], repeat=len(steps)))
        all_poss_seqs.sort(key=sum)  # sorting by sum means that we try the smallest action sequence lengths first
        egraph_base = self.new_egraph()

        for seq_mask in all_poss_seqs:
            eg = egraph_base.clone()
            actions = list(itertools.compress(steps, seq_mask))  # mask
            for action in actions:
                stop_reason = eg.run([self.rewrite_rules[action]], iter_limit=1, node_limit=self.node_limit)
                if stop_reason != 'NODE_LIMIT':
                    cost, ex = eg.extract(self.expr)
                    if cost == best_possible_cost:
                        return actions  # found the shortest action sequence to achieve cost equiv to egg











    def optimize(self, max_steps=500):
        """Attempt to extract the minimum sequence of actions needed to reach the cost found by egg."""
        best_possible_cost, best_possible_expr = self.exhaustive_search()
        print("egg cost:", best_possible_cost, "expr", best_possible_expr)
        egraph = self.new_egraph()
        prev_cost, _ = egraph.extract(self.expr)
        orig_cost = prev_cost
        steps = []
        for i in range(max_steps):
            action, stop_reason, cost = self.step(egraph, np.random.randint(0, len(self.rewrite_rules)))
            if stop_reason != 'SATURATED':  # if 'SATURATED', the step didn't change the egraph; we can filter it
                steps.append(Step(action, cost))
                if cost <= best_possible_cost:
                    print("found best cost at", len(steps) - 1, "cost", cost)
                    break  # we don't need to take any more steps

        print("Steps")
        print(steps)
        # Build a list of 'sentinel' actions: these are actions which directly reduce the extraction cost.
        sentinel_actions = []
        for ind, (action, cost) in enumerate(steps):
            if cost < prev_cost:
                prev_cost = cost
                sentinel_actions.append(SentinelStep(action, cost, ind))

        print("Sentinel Actions")
        print(sentinel_actions)
        # For each pair of sentinel steps, find the minimum set of intermediary actions to get us from one to the other
        sentinel_step_pairs = [(sentinel_actions[i], sentinel_actions[i + 1]) for i in range(len(sentinel_actions) - 1)
                               # filter out adjacent pairs
                               if sentinel_actions[i].step_ind != sentinel_actions[i+1].step_ind - 1]

        if sentinel_actions[0].step_ind != 0:
            # First sentinel isn't first action. So we need to consider actions [0, first_sentinel] as a pair
            sentinel_step_pairs.insert(0, (SentinelStep(-1, orig_cost, -1), sentinel_step_pairs[0][0]))

        if len(sentinel_step_pairs) == 0:
            action_sequence = [a.action for a in sentinel_actions]
            print("Ended early, every sentinel was adjacent")
            print(action_sequence)
            return

        action_sequence = []
        print("Sentinel pairs:")
        print(sentinel_step_pairs)
        for pair in sentinel_step_pairs:
            action_sequence += self.solve_gen_simple(pair, steps)
        action_sequence.append(sentinel_step_pairs[-1][1].action)

        print("Built minimum action sequence")
        print(action_sequence)

        # Check that we get the same result
        egraph = self.new_egraph()
        for a in action_sequence:
            egraph.run([self.rewrite_rules[a]], iter_limit=1, node_limit=self.node_limit)
        cost, new_expr = egraph.extract(self.expr)
        print("final cost", cost, "expr", new_expr)
        assert cost == best_possible_cost


    def solve_gen_simple(self, sentinel_pair: tuple[SentinelStep, SentinelStep], steps: list[Step]) -> list[Step]:
        head, tail = sentinel_pair
        print("Optimizing", head, tail)
        possible_actions = [s.action for s in steps[head.step_ind+1:tail.step_ind]]
        egraph_at_start = self.build_egraph_to_step_ind(steps, head.step_ind)
        min_num_actions = len(possible_actions)
        best_seq = []

        for action_seq in itertools.product([0, 1], repeat=len(possible_actions)):
            # egraph = self.build_egraph_to_step_ind(steps, head.step_ind)
            egraph = egraph_at_start.clone()
            actions_to_apply = [possible_actions[ind] for ind, mask in enumerate(action_seq) if mask == 1]
            num_actions = len(actions_to_apply)
            for i in range(num_actions):
                egraph.run([self.rewrite_rules[actions_to_apply[i]]], iter_limit=1, node_limit=self.node_limit)
            egraph.run([self.rewrite_rules[tail.action]], iter_limit=1, node_limit=self.node_limit)
            cost, ex = egraph.extract(self.expr)
            print(action_seq, "cost", cost, "goal cost", tail.cost, expr)
            if cost != tail.cost:
                # this sequence doesn't get us to the tail cost, so discard it
                continue
            else:
                # this is a potentially valid solution. But is it the shortest?
                if num_actions < min_num_actions:
                    min_num_actions = num_actions
                    best_seq = actions_to_apply

        if head.action >= 0:
            return [head.action] + best_seq
        else:
            return best_seq

    def solve_gen(self, sentinel_pair: tuple[SentinelStep, SentinelStep], steps: list[Step]) -> list[Step]:
        head, tail = sentinel_pair
        possible_actions = [s.action for s in steps[head.step_ind+1:tail.step_ind]]

        def fitness(sol: list[int], _solution_ind):
            egraph = self.build_egraph_to_step_ind(steps, head.step_ind)
            actions_to_apply = [possible_actions[ind] for ind, mask in enumerate(sol) if mask == 1]
            num_actions = len(actions_to_apply)
            for i in range(num_actions):
                egraph.run([self.rewrite_rules[actions_to_apply[i]]], iter_limit=1, node_limit=self.node_limit)
            cost, ex = egraph.extract(self.expr)
            fitness_val = (1.0 / np.abs(cost - tail.cost)) - num_actions
            print("sol", sol, "cost", cost, "expr", ex, "fit", fitness_val)
            return fitness_val

        ga = pygad.GA(num_generations=10,
                      num_parents_mating=5,
                      mutation_num_genes=1,
                      fitness_func=fitness,
                      sol_per_pop=20,
                      init_range_low=0,
                      init_range_high=1,
                      gene_type=int,
                      gene_space=[0, 1],
                      num_genes=len(possible_actions))
        ga.run()
        solution, solution_fitness, solution_idx = ga.best_solution()
        return [head.action] + [possible_actions[ind] for ind, mask in enumerate(solution) if mask == 1]

    def build_egraph_to_step_ind(self, steps: list[Step], ind: int):
        egraph = self.new_egraph()
        for i in range(ind + 1):
            egraph.run([self.rewrite_rules[steps[i].action]], 1)
        return egraph

    def step(self, egraph, action: int):
        rewrite_to_apply = [self.rewrite_rules[action]]
        stop_reason = egraph.run(rewrite_to_apply, iter_limit=1, node_limit=self.node_limit)
        best_cost, best_expr = egraph.extract(self.expr)
        best_cost = float(best_cost)
        return action, stop_reason, best_cost

    def exhaustive_search(self, iter_limit=7):
        egraph = self.new_egraph()
        egraph.run(lang.rewrite_rules(), iter_limit=iter_limit, node_limit=self.node_limit)
        best_cost, best_expr = egraph.extract(expr)
        return best_cost, best_expr

    def new_egraph(self):
        egraph = EGraph()
        egraph.add(self.expr)
        return egraph


if __name__ == "__main__":
    lang = TestLang()
    ops = lang.all_operators_obj()
    expr = ops.mul(1, ops.add(7, ops.mul(ops.add(16, 2), ops.mul(4, 0))))
    solver = GeneticSolver(lang=lang, expr=expr)
    solver.prim_optimize()
