import multiprocessing as mp

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo
import torch
import yaml
from scipy import stats
from tqdm import tqdm
from velotest.hypothesis_testing import run_hypothesis_test_on
from velotest.neighbors import find_neighbors, find_neighbors_in_direction_of_velocity


def calculateBound(adata, k=None, neighbors=True, num_neighbors=200): 
    if neighbors is True:
        nn_indices_optimal = find_neighbors(torch.tensor(adata.layers['Ms']), k_neighbors=num_neighbors)

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
            maxs[i] = torch.max(cosines)
        else:
            maxs[i] = torch.mean(torch.topk(cosines, largest=True, k=k).values)

    return maxs

def bestVelocityEmbedded(adata, basis, number_neighbors):
    # Run test - tsne
    uncorrected_p_values, h0_rejected, debug_dict = run_hypothesis_test_on(adata, basis=basis, number_neighbors_to_sample_from=number_neighbors, null_distribution="velocities-explicit", cosine_empty_neighborhood=None, correction="bonferroni", restrict_to_velocity_genes=True, exclusion_degree=10, n_workers=18)
    statistics = debug_dict['statistics']

    # Calculate emmbeding test optimal statistic
    best_value = [statistics.get_max_value(offset=True)[1] for statistics in statistics if statistics is not None]
    non_empty_neighborhoods_indices = np.where(uncorrected_p_values != 2)[0]

    return best_value, non_empty_neighborhoods_indices

def changeTSNE(adata, seed, perplexity=30):
    if 'X_tsne' in adata.obsm:
        del adata.obsm['X_tsne'], adata.obsm['velocity_tsne'], adata.obsp['connectivities'], adata.obsp['distances']
    sc.tl.tsne(adata, random_state=seed, perplexity=perplexity)
    sc.pp.neighbors(adata) 
    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_embedding(adata, basis='tsne')
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
        #"RPE_1": "configs/noEmbedding/intestinal_organoids.yaml",
        #"Hindbrain_mouse": "configs/noEmbedding/hindbrain.yaml",
        #"Oligodendrocyte_mouse": "configs/noEmbedding/oligodendrocyte.yaml",
        #"Bone_marrow_mouse": "configs/noEmbedding/mouse_bonemarrow.yaml",
        #"Cerebral_cortex_human": "configs/noEmbedding/human_cerebral.yaml",
        #"Hematopoiesis_mouse": "configs/noEmbedding/mouse_hema.yaml",
        #"Spermatogenisis": "configs/noEmbedding/spermatogenesis.yaml",
        #"Old_brain": "configs/noEmbedding/old_brain.yaml",
        #"Prefrontal_cortex": "configs/noEmbedding/pfc.yaml",
        #"Intestinal_organoids":"configs/noEmbedding/GSE128365.yaml",
        "Zebra_NMP": "/mnt/volume/Johanna_Velotest/configs/noEmbedding/zebra_NMP.yaml"
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
  

        # Prepare subsampling 
        cells = 2000
        neighbors = 100
        model = config['model']

        distances = []
        for seed in range(25):
            print(f"Main script: {seed}")
            if (adata.n_obs < cells):
                print(f"Dataset {name} has less than {cells} cells, skipping.")
                continue
            # Subsample dataset
            adata_frac = sc.pp.sample(adata, n=cells, axis='obs', copy=True)
            scvelo.pp.filter_and_normalize(adata_frac, n_top_genes=2000)
            scvelo.pp.moments(adata_frac, n_pcs=30, n_neighbors=30)
            scvelo.tl.recover_dynamics(adata_frac, n_jobs=7)
            scvelo.tl.velocity(adata_frac, mode=model)

            upper_optimal_embedding = calculateBound(adata_frac, neighbors=True, num_neighbors=neighbors)

            # Recalculate embedding and distance
            adata_frac = changeTSNE(adata_frac, seed)
            # Added this debugging lines only for the PBMC dataset
            if np.isnan(adata_frac.obsm['velocity_tsne']).any():
                valid_embedding_mask = ~np.isnan(adata_frac.obsm['velocity_tsne']).any(axis=1)
                adata_frac = adata_frac[valid_embedding_mask].copy()
                # Report removed cells
                removed_count = len(valid_embedding_mask) - valid_embedding_mask.sum()
                print(f"Removed {removed_count} cells with broken embeddings.")
            best, non_empty_neighborhoods_indices = bestVelocityEmbedded(adata_frac, basis="tsne", number_neighbors=neighbors)
            distance = stats.wasserstein_distance(best, upper_optimal_embedding)
            distances.append(distance)
        # Save distances
        if not distances:
            results[name] = [np.nan]
        else:
            results[name] = distances
            save = pd.DataFrame(distances)
            save.to_csv(f"results/distances/tsne/{dataset_name}.csv", index=False)
    df = pd.DataFrame(results)
    df.to_csv("results/distances/0616_Was_all_subsampled_2000cellsZebra.csv", index=False)
    print("Done! Results saved.")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        main()
    except RuntimeError:
        pass
