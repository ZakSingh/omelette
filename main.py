from MathLang import MathLang
from rejoice import EGraph, ppo_model, networks, envs
import torch_geometric as geom
from pytorch_lightning import Trainer, loggers
import os


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


def main():
    lang = MathLang()
    ops = lang.all_operators_obj()
    expr = ops.mul(ops.add(ops.add(0, 1), 1), 0)
    egraph = EGraph()

    logdir = os.getcwd()
    tb_logger = loggers.TensorBoardLogger(logdir)

    model = ppo_model.PPO(env="egraph", lang=lang, egraph=egraph, expr=expr)
    trainer = Trainer(max_epochs=15, log_every_n_steps=5, logger=tb_logger)
    trainer.fit(model)


if __name__ == "__main__":
    main()
