import argparse
import time
import os
from os import listdir
import random
import re
from collections import namedtuple
import torch
import numpy as np
import pandas as pd
from MathLang import MathLang
from PropLang import PropLang
from ppo import run_ppo
from rejoice.lib import Language
from rejoice.rejoice import EGraph

Step = namedtuple("Step", ['action', 'action_name', 'stop_reason', 'cost', 'num_applications', 'num_enodes', 'num_eclasses', 'best_expr', 'init_expr'])

default_out_path = "dataset_metrics"

def new_egraph(expr):
    egraph = EGraph()
    egraph.add(expr)
    return egraph

def base_cost(expr):
    """Get the cost of the root expression in the initial egraph"""
    egraph = EGraph()
    egraph.add(expr)
    best_cost, _ = egraph.extract(expr)
    return best_cost

def step(action: int, expr_to_extract, lang: Language, egraph: EGraph, node_lim=10_000):
    rw_rules = lang.rewrite_rules()
    rewrite_to_apply = [rw_rules[action]]
    stop_reason, num_applications, num_enodes, num_eclasses = egraph.run(rewrite_to_apply, iter_limit=1, node_limit=node_lim)
    best_cost, best_expr = egraph.extract(expr_to_extract)
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

def add_df_meta(df: pd.DataFrame, lang_name: str, solver_name: str, training_time=0.0):
    df["lang"] = lang_name
    df["solver"] = solver_name
    df["training_time"] = training_time
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

def rollout(lang: Language, expr, device, agent, training_time, num_rollouts=100, max_ep_len=100):
    """Rollout an agent's trained policy on a given expression."""

    def run_once(lang, expr, device, agent, max_ep_len=100):
        egraph = new_egraph(expr)
        steps = []
        agent.eval()
        count = 0
        while True:
            obs = lang.encode_egraph(egraph, use_shrink_action=True, step=count).to(device)
            with torch.no_grad():
                action, *rest = agent.get_action_and_value(obs, invalid_action_mask=obs.action_mask)

            action = action.item()
            if action == lang.num_rules:
                print("end action received", action, "at", count)
                break  # network has told us to take the end action
            elif action == lang.num_rules + 1:
                print("rebase action received", action, "at", count)
                _, expr = egraph.extract(expr)
                egraph = new_egraph(expr)
                # TODO: Append "rebase" action to step list
            else:
                s = step(action, expr, lang, egraph, node_lim)
                steps.append(s)
                if s.stop_reason == 'NODE_LIMIT' or s.stop_reason == 'TIME_LIMIT':
                    print("node or time limit hit during policy rollout")
                    break  # should be rare
                if count >= max_ep_len:
                    break
            count += 1

        if len(steps) == 0:
            return None

        stepdf = pd.DataFrame(steps)
        return stepdf


   # PPO policy is stochastic, so try multiple times

    best_rollout_cost = np.inf
    best_rollout_len = np.inf
    best_rollout = None

    for i in range(num_rollouts):
        steps_df = run_once(lang, expr, device, agent, max_ep_len)
        if steps_df is None or len(steps_df) == 0:
            continue  # rollout ended immediately

        cost = steps_df['cost'].iloc[-1]
        num_steps = len(steps_df)

        if num_steps <= best_rollout_len and cost <= best_rollout_cost:
            best_rollout = steps_df
            print('new best', "c", cost, "s", num_steps, 'old', "c", best_rollout_cost, "s", best_rollout_len)
            best_rollout_cost = cost
            best_rollout_len = num_steps

    best_rollout = add_df_meta(best_rollout, lang.name, "omelette", training_time=training_time)
    return best_rollout


def solve_expr_omelette(lang: Language, expr, expr_ind: int, egg_cost: int, egg_expr: str, node_lim=10_000, num_rollouts=100, max_ep_len=10, seed=1):
    """Train the PPO agent with its default config on this single expression in isolation."""
    print("Training agent...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_stamp = int(round(time.time() * 1000))
    agent = run_ppo(lang=lang.name,
                    seed=seed,
                    expr_str=str(expr),
                    exp_name=f"{lang.name}_{expr_ind}", 
                    node_limit=node_lim,
                    use_shrink_action=True,
                    learning_rate=5e-4,
                    max_episode_steps=max_ep_len,
                    total_timesteps=100_000,
                    egg_cost=egg_cost,
                    egg_expr=egg_expr,
                    max_cost=base_cost(expr),
                    print_actions=False)
    second_stamp = int(round(time.time() * 1000))
    training_time = second_stamp - first_stamp
    print("Agent trained. Evaluating learned policy...")

    df = rollout(lang=lang,
                 expr=expr,
                 device=device,
                 agent=agent,
                 training_time=training_time,
                 num_rollouts=num_rollouts,
                 max_ep_len=max_ep_len)
    return df

def solve_expr(lang: Language, expr, expr_ind: int, node_lim=10_000, seed=1, out_path=default_out_path):
    print("Solving expression", expr)

    egg_df = solve_expr_egg(lang, expr, node_lim)
    print("egg cost:", egg_df["cost"].iloc[-1])
    egg_df.to_feather(f"{out_path}/{lang.name}_{expr_ind}_egg")

    om_df = solve_expr_omelette(lang=lang,
                                expr=expr,
                                expr_ind=expr_ind,
                                max_ep_len=100,
                                node_lim=node_lim,
                                egg_cost=egg_df["cost"].iloc[-1],
                                egg_expr=egg_df["best_expr"].iloc[-1])
    om_df.to_feather(f"{out_path}/{lang.name}_{expr_ind}_om")


def om_multitask(lang: Language, exprs: list, envs_per_expr=8, node_lim=10_000, num_eval_rollouts=100, max_ep_len=100, seed=1):
    """
    Run omelette in mult-task configuration. This involves supplying the agent with 8 of each expr.
    """
    expr_strs = [str(expr) for _, expr in exprs]
    exp_names = [f"{lang.name}_{expr_ind}" for expr_ind, _ in exprs]
    exps = list(zip(exp_names, expr_strs))

    agent = run_ppo(lang=lang.name,
                    expr_list=exps,
                    node_limit=node_lim,
                    seed=seed,
                    use_shrink_action=True,
                    learning_rate=5e-4,
                    max_episode_steps=max_ep_len,
                    total_timesteps=100_000,
                    print_actions=False)
    
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
    # exprs = [exprs[2]]
    # exprs = [(0, lang.get_single_task_exprs().saturatable)]

    # filter expressions we already have in output dir
    # already_done_inds = [int(re.search(f'{lang.name}_(.+?)', file).group(1)) for file in listdir(out_path)]
    # print("already done", already_done_inds)
    # exprs = [i for j, i in enumerate(exprs) if j not in already_done_inds]

    for expr_ind, expr in exprs:
        try:
            solve_expr(lang=lang, expr_ind=expr_ind, expr=expr, node_lim=node_lim, out_path=out_path, seed=seed)
        except:
            print("Failed to solve expr_ind", expr_ind)

    print("Completed running all experiments in generated dataset.")


if __name__ == "__main__":
    node_lim = 500
    run_exps("PROP", num_expr=100, node_lim=node_lim, seed=1)