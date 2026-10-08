import multiprocessing as mp
import yaml

import scvelo
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from scipy import stats
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm

import velotest
from velotest.hypothesis_testing import run_hypothesis_test_on
from velotest.neighbors import find_neighbors_in_direction_of_velocity, find_neighbors


def calculateBound(adata, k=None, neighbors=True, num_neighbors=200): 
    if neighbors is True:
        nn_indices_optimal = find_neighbors(torch.tensor(adata.layers['Ms']), k_neighbors=num_neighbors)

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

def bestVelocityEmbedded(adata, basis, number_neighbors):
    # Run test - tsne
    uncorrected_p_values, h0_rejected, debug_dict = run_hypothesis_test_on(adata, basis=basis, 
                                                                           number_neighbors_to_sample_from=number_neighbors, 
                                                                           null_distribution="velocities-explicit", 
                                                                           cosine_empty_neighborhood=None, correction="bonferroni", 
                                                                           restrict_to_velocity_genes=True, exclusion_degree=10, n_workers=18)
    statistics = debug_dict['statistics']

    # Calculate emmbeding test optimal statistic
    best_value = [statistics.get_max_value(offset=True)[1] for statistics in statistics if statistics is not None]
    non_empty_neighborhoods_indices = np.where(uncorrected_p_values != 2)[0]

    return best_value, non_empty_neighborhoods_indices

def changeUMAP(adata, seed, neighbors=30):
    del adata.obsm['X_umap'], adata.obsm['velocity_umap'], adata.obsp['connectivities'], adata.obsp['distances']
    sc.pp.neighbors(adata, n_neighbors=neighbors, random_state=seed)  # recompute graph
    sc.tl.umap(adata, init_pos='random', random_state=seed)
    scvelo.pl.velocity_embedding(adata)

    return adata

def changeTSNE(adata, seed, perplexity=30):
    if 'X_tsne' in adata.obsm:
        del adata.obsm['X_tsne'], adata.obsm['velocity_tsne']
    sc.tl.tsne(adata, random_state=seed, perplexity=perplexity)
    scvelo.pl.velocity_embedding(adata, basis='tsne')
    return adata

def main():
    # Load preprocessed dataset
    datasets = {
        #"Pancreas_dynamical": "configs/pancreas_dynamic.yaml",
        #"Pancreas_stochastic": "configs/pancreas_stochastic.yaml",
        #"Dentate_gyrus": "configs/dentategyrus.yaml",
        #"Gastrulation_erythroid": "configs/gastrulation.yaml",
        #"Bone_marrow": "configs/bonemarrow.yaml",
        #"PBMC": "configs/pbmc.yaml",
        #"Covid": "configs/covid.yaml",
        #"Retinal_development": "configs/retinal_development.yaml",
        #"Human_HSC": "configs/humanHSC.yaml",
        #"Intestinal_development": "configs/noEmbedding/intestinal_organoids.yaml",
        #"Hindbrain_mouse": "configs/noEmbedding/hindbrain.yaml",
        #"Oligodendrocyte_mouse": "configs/noEmbedding/oligodendrocyte.yaml",
        #"Bone_marrow_mouse": "configs/noEmbedding/mouse_bonemarrow.yaml",
        #"Cerebral_cortex_human": "configs/noEmbedding/human_cerebral.yaml",
        #"Hematopoiesis_mouse": "configs/noEmbedding/mouse_hema.yaml",
        #"Spermatogenisis": "configs/noEmbedding/spermatogenesis.yaml",
        #"Old_brain": "configs/noEmbedding/old_brain.yaml",
        #"Prefrontal_cortex": "configs/noEmbedding/pfc.yaml",
        #"Intestinal_organoids":"configs/noEmbedding/GSE128365.yaml",
        "Zebra_NMP": "configs/noEmbedding/zebra_NMP.yaml"
    }

    results = {}

    # Compute Wasserstein Disntance for each dataset
    for name, path in datasets.items():
        print(f"Processing dataset: {name}")
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        # Load data from config file
        adata = sc.read(config['path'])
        dataset_name = config['dataset_name']
        neighbors = config['number_neighbors_to_sample_from']   


        upper_optimal_embedding = calculateBound(adata, neighbors=True, num_neighbors=neighbors)
        distances = []
        for seed in range(25):
            print(f"Main script: Start change Embedding for seed {seed}")
            adata = changeTSNE(adata, seed, perplexity=50)
            best, non_empty_neighborhoods_indices = bestVelocityEmbedded(adata, basis="tsne", number_neighbors=neighbors)
            distance = stats.wasserstein_distance(best, upper_optimal_embedding)
            distances.append(distance)
        results[name] = distances
        save = pd.DataFrame(distances)
        #save.to_csv(f"results/umap/{dataset_name}.csv", index=False)
    df = pd.DataFrame(results)
    df.to_csv("results/distances/0525_Was_zebra_NMP_tsne_p50.csv", index=False)
    print("Done! Results saved.")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        main()
    except RuntimeError:
        pass
