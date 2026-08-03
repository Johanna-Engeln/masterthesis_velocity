import multiprocessing as mp

import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo
import seaborn as sns
import torch
import velotest
import yaml
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
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
        del adata.obsm['X_tsne'], adata.obsm['velocity_tsne']
    sc.tl.tsne(adata, random_state=seed, perplexity=perplexity)
    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_embedding(adata, basis='tsne')
    return adata

def plot_streamplot(adata, basis, output_path, dataset_name, color, figsize=(5, 4)):
    # Start plot frame
    fig, ax = plt.subplots(dpi=450, figsize=figsize)
    ax.set(xticks=[], yticks=[])
   
    # Plot original streamplot
    pl = scvelo.pl.velocity_embedding_stream(adata, basis=basis, 
                                             color=color,
                                             title='', frameon=False,
                                             ax=ax,  xlabel='', ylabel='', 
                                             legend_fontsize=5, legend_fontweight='normal', size=10, alpha=0.9,
                                             linewidth=.25, arrow_size=0.25, arrow_color='black', edgecolors=None,
                                             rasterized=True, figsize=figsize, 
                                             dpi=450, show=False, add_margin = 0.07)
    ax.set_aspect('equal')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    pl.set_rasterized(True)

    # Save plot
    plt.savefig(f"{output_path}/{dataset_name}_{basis}_streamplot.pdf",
        transparent=True, dpi=450)
    plt.close()
    pass

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
        "RPE_1": "configs/noEmbedding/intestinal_organoids.yaml",
        "Hindbrain_mouse": "configs/noEmbedding/hindbrain.yaml",
        "Oligodendrocyte_mouse": "configs/noEmbedding/oligodendrocyte.yaml",
        "Bone_marrow_mouse": "configs/noEmbedding/mouse_bonemarrow.yaml",
        "Cerebral_cortex_human": "configs/noEmbedding/human_cerebral.yaml",
        "Hematopoiesis_mouse": "configs/noEmbedding/mouse_hema.yaml",
        "Spermatogenisis": "configs/noEmbedding/spermatogenesis.yaml",
        "Old_brain": "configs/noEmbedding/old_brain.yaml",
        "Prefrontal_cortex": "configs/noEmbedding/pfc.yaml",
        "Intestinal_organoids":"configs/noEmbedding/GSE128365.yaml",
        "Zebra_NMP": "/mnt/volume/Johanna_Velotest/configs/noEmbedding/zebra_NMP.yaml"
    }

    results = []

    # Compute Wasserstein Disntance for each dataset
    for name, path in datasets.items():
        print(f"Processing dataset: {name}")
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        # Load data from config file
        adata = sc.read(config['path'])
        dataset_name = config['dataset_name']
        model = config['model']
        cluster_color = config['cluster_colors']
  
        # Find a good cluster to subsample from
        cluster = config['cluster']
        labels = adata.obs[cluster]
        number_cluster = len(labels.unique())
        order_clusters = labels.value_counts()
        middle_idx = (number_cluster -1) // 2
        while order_clusters.iloc[middle_idx] < 100 and middle_idx > 0:
            middle_idx -= 1

        middle_cluster = order_clusters.index[middle_idx]
        middle_cluster_size = order_clusters.iloc[middle_idx]
        if name == 'RPE_1':
            middle_cluster = order_clusters.index[1]
            middle_cluster_size = order_clusters.iloc[1]
        print(f"Total clusters: {number_cluster}")
        print(order_clusters)
        print(f"Selected middle index: {middle_idx} | Cluster name: {middle_cluster} | Size: {middle_cluster_size}")

        high_cluster = order_clusters.index[0]
        high_cluster_size = order_clusters.iloc[0]
        print(f"Selected high index: {middle_idx} | Cluster name: {high_cluster} | Size: {high_cluster_size}")

        # Subsample cluster
        adata_frac = adata[adata.obs[cluster] == high_cluster].copy()  # Swap this line to check a different cluster
        if name == "Human_HSC":
            adata_frac.X = adata_frac.layers['matrix']
        scvelo.pp.filter_and_normalize(adata_frac, n_top_genes=2000)
        sc.tl.pca(adata_frac)
        sc.pp.neighbors(adata_frac, n_pcs=30, n_neighbors=30) 
        scvelo.pp.moments(adata_frac)
        if model == 'dynamical':
            scvelo.tl.recover_dynamics(adata_frac, n_jobs=7)
        scvelo.tl.velocity(adata_frac, mode=model)

        neighbors = int(high_cluster_size * 0.05)
        upper_optimal_embedding = calculateBound(adata_frac, neighbors=True, num_neighbors=neighbors)
        
        distances = []
        for seed in range(25):
            print(f"Main script: {seed}")

            # Recalculate embedding and distance
            adata_frac = changeTSNE(adata_frac, seed)
            plot_streamplot(adata_frac, 'tsne', '/mnt/volume/Johanna_Velotest/results/distances/cluster_subsample', f'{dataset_name}_{seed}', cluster)
           
            best, non_empty_neighborhoods_indices = bestVelocityEmbedded(adata_frac, basis="tsne", number_neighbors=neighbors)
            distance = stats.wasserstein_distance(best, upper_optimal_embedding)
            distances.append(distance)

        number_velocity_genes = adata_frac.var['velocity_genes'].sum()
        data = {
            'dataset_name': dataset_name,
            'cluster_name': high_cluster,
            'cluster_size': high_cluster_size,
            'velocity_genes': number_velocity_genes,
            'distances': distances if distances else [np.nan]
        }
        results.append(data)

        df = pd.DataFrame(results)
        df.to_csv('/mnt/volume/Johanna_Velotest/results/distances/cluster_subsample/Wasserstein_tSNE_high_cluster.csv', index=False)
    print("Done! Results saved.")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        main()
    except RuntimeError:
        pass
