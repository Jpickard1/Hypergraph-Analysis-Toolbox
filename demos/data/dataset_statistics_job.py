import os
import sys
import json
from HAT import datasets, Hypergraph

download_path = "/nfs/turbo/umms-indikar/Joshua/hif-tests/data"
output_path = "/nfs/turbo/umms-indikar/Joshua/hif-tests/results"

def main(job_number):
    job_number = int(job_number)

    # Download and load dataset
    hif_datasets = datasets.list_datasets()
    dataset_name = hif_datasets[job_number]
    print(f"{job_number=}")
    print(f"{dataset_name=}")
    HG = datasets.load(dataset_name, datapath=download_path)

    # Hypergraph statistics
    n = HG.nodes.shape[0]
    e = HG.edges.shape[0]
    HG.edges['string_nodes'] = HG.edges['Nodes'].apply(lambda nodes: '|'.join(sorted(map(str, nodes))))
    unique_e = HG.edges['string_nodes'].nunique()

    print(f"{n=}")
    print(f"{e=}")
    print(f"{unique_e=}")
    
    # Get largest cc
    cc = HG.connected_components
    largest_cc = cc['connected components'].mode().values[0]
    nodes = HG.nodes[HG.nodes["connected components"] == largest_cc]['Nodes'].values
    reduced_im = HG.incidence_matrix[nodes, :]
    nonzero_cols = reduced_im.sum(axis=0) != 0  # Boolean mask
    reduced_im_final = reduced_im[:, nonzero_cols]

    # Largest hypergraph cc statistics
    HG_cc = Hypergraph(incidence_matrix = reduced_im_final)
    cc_n = HG_cc.nodes.shape[0]
    cc_e = HG_cc.edges.shape[0]
    HG_cc.edges['string_nodes'] = HG_cc.edges['Nodes'].apply(lambda nodes: '|'.join(sorted(map(str, nodes))))
    cc_unique_e = HG_cc.edges['string_nodes'].nunique()
    
    print(f"{cc_n=}")
    print(f"{cc_e=}")
    print(f"{cc_unique_e=}")

    # Pack results into a dict for serialization
    result = {
        "dataset": dataset_name,
        "n_nodes": n,
        "n_edges": e,
        "unique_edge_nodes": unique_e,
        "cc_n_nodes": cc_n,
        "cc_n_edges": cc_e,
        "cc_unique_edge_nodes": cc_unique_e
    }

    # Save result to disk (e.g., using job_number for filename)
    out_file = os.path.join(output_path, f"stats_{job_number}_{dataset_name}.json")
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved stats for {dataset_name} to {out_file}")

if __name__ == "__main__":
    job_number = sys.argv[1]  # get the first argument from command line
    main(job_number)
