from MathLang import MathLang
from rejoice import EGraph
import networkx as nx
import matplotlib
import torch_geometric as geom
import matplotlib.pyplot as plt


def main():
    lang = MathLang()

    ops = lang.all_operators_obj()
    expr = ops.add(ops.add(0, 1), 1)
    egraph = EGraph(lang.eclass_analysis)
    egraph.add(expr)
    egraph.rebuild()

    x, edge_index = lang.encode_egraph(egraph)
    data = geom.data.Data(x=x, edge_index=edge_index)
    g = geom.utils.to_networkx(data, node_attrs=['x'])

    for u, data in g.nodes(data=True):
        decoded = lang.decode_node(data["x"])
        if decoded["type"] == "eclass":
            data['name'] = "eclass"
        elif decoded["is_scalar"]:
            data['name'] = decoded["value"]
        else:
            data["name"] = decoded["op"]
        del data['x']

    node_labels = {}
    for u, data in g.nodes(data=True):
        node_labels[u] = data['name']

    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    nx.draw(g, labels=node_labels, pos=pos)
    plt.show()


if __name__ == "__main__":
    main()
