from HAT import Hypergraph
import hypernetx as hnx

def to_hypernetx(HG):
    HG_df = HG.edges.explode('Nodes')
    print(f"{HG_df=}")
    HG_df['Edges'] = HG_df.index
    print(f"{HG_df=}")
    return hnx.Hypergraph(HG_df, edge_col='Edges', node_col='Nodes')
