import multiprocessing as mp
import yaml

import scvelo
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from scipy import stats

from matplotlib import pyplot as plt
from tqdm import tqdm

import velotest
from velotest.hypothesis_testing import run_hypothesis_test_on
from velotest.neighbors import find_neighbors


#############################################################################
# Methods part 
##############################################################################
def calculateBound(adata, k=None, neighbors=True, number_neighbors=200): 
    if neighbors is True:
        nn_indices_optimal = find_neighbors(torch.tensor(adata.layers['Ms']), k_neighbors=number_neighbors)

    mins = torch.tensor([-2.]*len(adata))
    maxs = torch.tensor([-2.]*len(adata))
    for i in tqdm(range(len(adata))):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        if neighbors is True:
            expression_all = torch.tensor(adata.layers['Ms'][nn_indices_optimal[i]][:,adata.var.velocity_genes])
        else:
            expression_all = torch.tensor(adata.layers['Ms'][:,adata.var.velocity_genes])
        expression_cell = torch.tensor(adata.layers['Ms'][i,adata.var.velocity_genes]).unsqueeze(0)
        velocity = torch.tensor(adata.layers['velocity'][i,adata.var.velocity_genes]).unsqueeze(0)
        cosines = cos(expression_all - expression_cell, velocity)
        cosines = torch.cat([cosines[:i], cosines[i+1:]])
        if k is None:
            mins[i] = torch.min(cosines)
            maxs[i] = torch.max(cosines)
        else:
            mins[i] = torch.mean(torch.topk(cosines, largest=False, k=k).values)
            maxs[i] = torch.mean(torch.topk(cosines, largest=True, k=k).values)

    return maxs

def bestVelocityEmbedded(adata, basis, num=200):
    # Run test - tsne
    uncorrected_p_values, h0_rejected, debug_dict = run_hypothesis_test_on(adata, 
                                                                           basis=basis, 
                                                                           number_neighbors_to_sample_from=num, 
                                                                           null_distribution="velocities-explicit", 
                                                                           cosine_empty_neighborhood=None, 
                                                                           correction="bonferroni", 
                                                                           restrict_to_velocity_genes=True, 
                                                                           exclusion_degree=10,
                                                                           n_workers=18)
    statistics = debug_dict['statistics']

    # Calculate emmbeding test optimal statistic
    best_value = [statistics.get_max_value(offset=True)[1] for statistics in statistics if statistics is not None]

    return best_value

#############################################################################
# MAIN part
#############################################################################

def main():
    # Load preprocessed dataset
    datasets = {
        "Pancreas_dynamical": "configs/pancreas_dynamic.yaml",
        "Pancreas_stochastic": "configs/pancreas_stochastic.yaml",
        "Dentate_gyrus": "configs/dentategyrus.yaml",
        "Gastrulation_erythroid": "configs/gastrulation.yaml",
        "Bone_marrow": "configs/bonemarrow.yaml",
        "PBMC": "configs/pbmc.yaml",
        "Covid": "configs/covid.yaml",
        "Retinal_development": "configs/retinal_development.yaml",
        "Human_HSC": "configs/humanHSC.yaml",
        "Intestinal_development": "configs/noEmbedding/intestinal_organoids.yaml",
        "Hindbrain_mouse": "configs/noEmbedding/hindbrain.yaml",
        "Oligodendrocyte_mouse": "configs/noEmbedding/oligodendrocyte.yaml",
        "Bone_marrow_mouse": "configs/noEmbedding/mouse_bonemarrow.yaml",
        "Cerebral_cortex_human": "configs/noEmbedding/human_cerebral.yaml",
        "Intestinal_organoids":"configs/noEmbedding/GSE128365.yaml"
    }

    # Initialte object to store results
    results = {}

    # Iterate through datasets
    for name, path in datasets.items():
        print(f"Processing dataset: {name}")
        # Open and collect data information
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        adata = sc.read(config["path"])
        basis = config['basis']
        cluster = config['cluster']
        model = config['model']
        #fractions = [0.5, 0.25, 0.125]
        cells = [2000]

        # Collect cluster fraction results and embeddings for each dataset
        cluster_fractions = {}
        cluster_fractions[f"{name}_original"] = adata.obs[cluster].value_counts()/adata.n_obs

        fig, ax = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
        scvelo.pl.scatter(adata, basis=basis, color=cluster, legend_loc='on data', ax=ax[0], title="original")

        for i, frac in enumerate(cells): # Here change fraction and cells in this loop
            # Sample from dataset and collect cluster fraction information
            adata_frac = sc.pp.sample(adata, n=cells, axis='obs', copy=True) # here otherwise fraction=frac
            cluster_fractions[f"{name}_{cells}"] = adata_frac.obs[cluster].value_counts()/adata_frac.n_obs

            scvelo.pp.filter_and_normalize(adata_frac, n_top_genes=2000)
            scvelo.pp.moments(adata_frac, n_pcs=30, n_neighbors=30)
            scvelo.tl.recover_dynamics(adata_frac, n_jobs=7)
            scvelo.tl.velocity(adata_frac, mode=model)  

            # Recompute embedding and neighbors for sampled dataset
            if basis == 'umap':
                sc.pp.neighbors(adata_frac)
                sc.tl.umap(adata_frac)
                scvelo.tl.velocity_graph(adata) 
                scvelo.tl.velocity_embedding(adata, basis='umap')
            if basis == 'tsne':
                sc.tl.tsne(adata_frac)
                scvelo.tl.velocity_graph(adata) 
                scvelo.tl.velocity_embedding(adata, basis='tsne')
            # Plot embedding for sampled dataset
            scvelo.pl.scatter(adata_frac, basis=basis, color=cluster, legend_loc='on data', ax=ax[i+1], title=f"{cells} cells")

            # Compute distance of neighbors now
            neighbors = [10, 25, 50, 100,200, 300, 400, 800, 1200, 1600, 2000] 
            distances = {}
            # Upper bound with differnt neighbors
            for i, num in enumerate(neighbors):
                # Check if neighbor doesn't exceed number of cells in dataset
                if (num+1000) >= adata_frac.n_obs:
                    continue

                print("Processing", name, num)
                upper_bound = calculateBound(adata_frac, neighbors=True, number_neighbors=num)
                bound = bestVelocityEmbedded(adata_frac, basis, num)
                distances[f"n_{num}"] = stats.wasserstein_distance(upper_bound, bound)
            # Store distances for each fraction of the dataset
            results[f"{name}_{cells}"] = distances

        # After all fraction save results of fractioning for each dataset
        cluster_fractions_df = pd.DataFrame(cluster_fractions)
        cluster_fractions_df.to_csv(f"/mnt/volume/Johanna_Velotest/results/neighbors/{name}_cluster_2000cells_recomputed.csv")
        plt.suptitle(f"{name}")
        plt.savefig(f"/mnt/volume/Johanna_Velotest/results/neighbors/{name}_2000cells_recomputed.png")
        plt.close()

    df = pd.DataFrame(results)
    df.to_csv("/mnt/volume/Johanna_Velotest/results/neighbors/k_sizes_subsample_recomputed_2000cells.csv")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        main()
    except RuntimeError:
        pass