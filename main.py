import tracemalloc

import torch
from MathLang import MathLang
from rejoice import EGraph, generator, envs
from pytorch_lightning import Trainer, loggers
import os
import gym
import logging

FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)


def test():
    lang = MathLang()

    ops = lang.all_operators_obj()
    expr = ops.mul(ops.add(ops.add(0, 1), 1), 0)

    egraph = EGraph()
    egraph.add(expr)

    data = lang.encode_egraph(egraph)
    g1 = lang.viz_egraph(data)
    print(egraph.classes())
    egraph.run(lang.rewrite_rules(), 5)
    best = egraph.extract(expr)
    print("best", best)
    print(egraph.classes())
    egraph.rebuild()
    print(egraph.classes())

    data = lang.encode_egraph(egraph)
    g2 = lang.viz_egraph(data)


def run_agent(agent_type="DQN"):
    lang = MathLang()
    ops = lang.all_operators_obj()
    add = ops.add
    mul = ops.mul
    expr = add(add(mul(16, 2), mul(4, 0)), 3)

    egraph = EGraph()
    egraph.add(expr)
    egraph.run(lang.rewrite_rules(), 7)
    best_cost, best_expr = egraph.extract(expr)
    egraph.graphviz("egg_best.png")
    print("egg best cost:", best_cost, "best expr: ", best_expr)
    # return
    logdir = os.getcwd()
    tb_logger = loggers.TensorBoardLogger(logdir)

    if agent_type == "DQN":
        env = gym.make('egraph-v0', lang=lang, expr=expr)
        print("num_node_features", env.num_node_features, "num_actions", env.action_space)

        model = DQN(env)
        rollout(env, model)
    elif agent_type == "DQN_LIGHTNING":
        model = DQNLightning(env="egraph-v0", lang=lang, expr=expr,
                             batch_size=128,
                             lr=1e-2,
                             gamma=0.99,
                             sync_rate=100,
                             replay_size=100000,
                             warm_start_size=20,
                             eps_last_frame=20,
                             eps_start=1.0,
                             eps_end=0.01,
                             episode_length=10,
                             warm_start_steps=20)

        trainer = Trainer(
            max_epochs=100_000,
            log_every_n_steps=10,
            logger=tb_logger,
            # val_check_interval=100,
        )

        trainer.fit(model)

    elif agent_type == "PPO":
        model = ppo_model.PPO(env="egraph-v0", lang=lang, expr=expr,
                              network_type="SAGE",
                              gamma=0.99,
                              lam=0.95,
                              lr_actor=3e-4,
                              lr_critic=1e-3,
                              max_episode_len=10,
                              batch_size=32,
                              steps_per_epoch=100,
                              nb_optim_iters=4,
                              clip_ratio=0.2)
        trainer = Trainer(max_epochs=1000, log_every_n_steps=5, logger=tb_logger)
        trainer.fit(model)


def generate():
    gen = generator.NAGOGenerator()
    x = torch.randn(128, 3, 32, 32, requires_grad=True)
    gen.forward(x)

    # Export the model
    torch.onnx.export(gen.model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})


def main():
    run_agent(agent_type="DQN")


if __name__ == "__main__":
    main()
