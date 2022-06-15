import argparse
import os
from os import listdir
import random
import re
from collections import namedtuple
from distutils.util import strtobool
import torch
import numpy as np
import pandas as pd
import torch_geometric as pyg
from LambdaLang import LambdaLang
from MathLang import MathLang
from PropLang import PropLang
from rejoice.lib import Language

from rejoice.pretrain_dataset_gen import EGraphSolver
from rejoice.rejoice import EGraph

Step = namedtuple("Step", ['action', 'action_name', 'stop_reason', 'cost', 'num_applications', 'num_enodes', 'num_eclasses', 'best_expr', 'init_expr'])

default_out_path = "dataset_metrics"

# Set a consistent seed, etc.

# 1. generate 100 expressions of a given language

# For each expression, use an egg-like solver to exhaustively apply rewrite rules
# track down the action taken and the number of applications at each step
# as well as number of enodes, number of eclasses, etc.
# this doesn't need to be pyg; not learning from this data. Just lists are sufficient.

def new_egraph(expr):
    egraph = EGraph()
    egraph.add(expr)
    return egraph

def step(action: int, expr_to_extract, lang: Language, egraph: EGraph, node_lim=10_000):
    rw_rules = lang.rewrite_rules()
    rewrite_to_apply = [rw_rules[action]]
    stop_reason, num_applications, num_enodes, num_eclasses = egraph.run(rewrite_to_apply, iter_limit=1, node_limit=node_lim)
    best_cost, best_expr = egraph.extract(expr_to_extract)
    best_cost = float(best_cost)
    return Step(action=action,
                action_name=lang.rule_names[action],
                num_applications=num_applications,
                stop_reason=stop_reason,
                cost=float(best_cost),
                best_expr=str(best_expr),
                num_eclasses=num_eclasses,
                num_enodes=num_enodes,
                init_expr=str(expr_to_extract)
                )

def add_df_meta(df: pd.DataFrame, lang_name: str, solver_name: str):
    df["lang"] = lang_name
    df["solver"] = solver_name
    # add the step index as a column
    df = df.reset_index().rename(columns={'index': 'step_ind'})
    return df

def solve_expr_egg(lang: Language, expr, node_lim=10_000):
    """
    Emulate egg's solver but WITHOUT an iteration limit.
    This will keep running until saturation, a node limit, or time limit is reached.
    """
    egraph = new_egraph(expr)
    steps = []

    i = 0
    sat_counter = 0

    while True:
        action_to_apply = i % lang.num_rules
        if action_to_apply == 0:
            sat_counter = 0

        result = step(action_to_apply, expr, lang, egraph, node_lim)
        steps.append(result)

        if result.stop_reason == 'NODE_LIMIT' or result.stop_reason == 'TIME_LIMIT':
            break  # egg stops optimizing
        elif result.stop_reason == 'SATURATED':
            sat_counter += 1

        if sat_counter == lang.num_rules:
            break  # egg has achieved saturation
        
        i += 1
    
    steps_df = pd.DataFrame(steps)
    steps_df = add_df_meta(steps_df, lang.name, "egg")

    return steps_df

def solve_expr_omelette(lang: Language, expr, node_lim=10_000):
    """Train the PPO agent with its default config on this expression."""
    pass

def solve_expr(lang: Language, expr, expr_ind: int, node_lim=10_000, out_path=default_out_path):
    egg_df = solve_expr_egg(lang, expr, node_lim)
    om_df = solve_expr_egg(lang, expr, node_lim)
    om_df["solver"] = "omelette"
    # omelette_df = solve_expr_omelette(lang, expr, node_lim)
    # requires training PPO agent, then keeping it around to execute converged to policy?
    # catastrophic forgetting would fuck with this badly.
    # need to run it, have the needed data be tracked during execution (w/ Tensorboard or w/o)
    # Add more tables that the PPO agent writes stuff into with each step.
    # Then on the episode_end, pull from those tables to get answers?

    expr_data = pd.concat([egg_df, om_df]).reset_index(drop=True)
    expr_data.to_feather(f"{out_path}/{lang.name}_{expr_ind}")
    
def get_lang(name: str) -> Language:
    return {
        "PROP": PropLang,
        "PropLang": PropLang,
        "MATH": MathLang,
        "MathLang": MathLang
    }[name]

def run_exps(lang_name: str, num_expr=10, node_lim=10_000, out_path=default_out_path, seed=1):
    # set random seeds for reproducability
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # create output dir if not exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    lang = get_lang(lang_name)()
    exprs = [(i, lang.gen_expr(p_leaf=0.0)) for i in range(num_expr)]

    # filter expressions we already have in output dir
    already_done_inds = [int(re.search(f'{lang.name}_(.+?)', file).group(1)) for file in listdir(out_path)]
    print("already done", already_done_inds)
    exprs = [i for j, i in enumerate(exprs) if j not in already_done_inds]
    print("exprs after filter", exprs)

    for expr_ind, expr in exprs:
        solve_expr(lang=lang, expr_ind=expr_ind, expr=expr, node_lim=node_lim, out_path=out_path)

    print("Completed running all experiments in generated dataset.")


if __name__ == "__main__":
    run_exps("PROP", num_expr=10, node_lim=10_000, seed=2)
    
    

# dataframe would have a column called 'input_expr' which is the string of the input expression so that we can group by task.
# then within that, each row has an 'index' so that ordering is tracked
# 

# 2. For each expression, train the PPO RL agent on the task.
# Need to somehow switch to 'eval' mode on the agent and see what actions it takes, run that
# policy 5 times, tracking the action and number of applications at each step, then taking the one that gives
# the lowest possible cost in the fewest actions.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    
    # omelette-specific configuration
    parser.add_argument("--mode", type=str, choices=["single_task_sat", "single_task_explodes", "bc", "multitask"], default="single_task_sat")
    parser.add_argument("--termination-decay", type=bool, default=True,
                        help="Prevent the agent from taking the termination action until n steps have elapsed.")

    parser.add_argument("--multitask-count", type=int, default=16,
                        help="the number of tasks to generate for multitask eval")
    parser.add_argument("--print-actions", type=bool, default=False,
                        help="print the (action, reward) tuples that make up each episode")
    parser.add_argument("--pretrained-weights-path", type=str, default=None,
                        help="Whether or not to pretrain the value and policy networks")
    parser.add_argument("--lang", type=str, default="PROP",
                        help="The language to use. One of PROP, MATH, TENSOR.")
    parser.add_argument("--use-action-mask", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, action masking is enabled")
    parser.add_argument("--use-edge-attr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, use edge attributes denoting difference between e-class member and e-node child edges")
    parser.add_argument("--use-shrink-action", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, include the shrink-and-reexpand action in the action-space")
    parser.add_argument("--node-limit", type=int, default=10_000,
                        help="egraph node limit")
    parser.add_argument("--num-egg-iter", type=int, default=7,
                        help="number of iterations to run egg for")
    parser.add_argument("--max-episode-steps", type=int, default=10,
                        help="the maximum number of steps in any episode")


    args = parser.parse_args()
    return args