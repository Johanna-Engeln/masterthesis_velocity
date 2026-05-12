import multiprocessing as mp
import yaml
import scanpy as sc
import scvelo
import torch
import numpy as np
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt

from velotest.explicit_hypothesis_testing import compute_p_values
from velotest.hypothesis_testing import run_hypothesis_test_on, correct_for_multiple_testing
from velotest.neighbors import find_neighbors, find_neighbors_in_direction_of_velocity
from velotest.plotting import arrow_plot, plot_uniformity_histogram, plot_best_possible_velocities_statistic, \
    plot_statistic_distribution, plot_neighborhood, compute_angle_on_gridplot_between

def neighborhoods(Z_expr, adata, cell, figsize, nn_indices, selected_neighbors, dataset_name, number_neighbors):
    _, ax = plt.subplots(dpi=450, constrained_layout=True, figsize=figsize)
    plot_neighborhood(cell, adata.obsm[Z_expr], nn_indices, ax=ax, s=0.3, vector_friendly=True,
                      selected_neighbors=selected_neighbors[cell])
    ax.set_aspect('equal')
    plt.savefig(
        f"results/neighbors/{dataset_name}-neighborhood_{cell}-{number_neighbors}.pdf",
        transparent=True, dpi=600)
    

def main():
    datasets = { #Change with comment function which datasets to run
            #"Pancreas_dynamical": "configs/pancreas_dynamic.yaml",
            #"Pancreas_stochastic": "configs/pancreas_stochastic.yaml",
            #"Dentate_gyrus": "configs/dentategyrus.yaml",
            #"Gastrulation_erythroid": "configs/gastrulation.yaml",
            #"Bone_marrow": "configs/bonemarrow.yaml",
            #"Covid": "configs/covid.yaml",
            "PBMC": "configs/pbmc.yaml",
            #"Retinal_development": "configs/retinal_development.yaml",
            #"Human_HSC": "configs/humanHSC.yaml",
            #"Intestinal_development": "configs/noEmbedding/intestinal_organoids.yaml",
            #"Hindbrain_mouse": "configs/noEmbedding/hindbrain.yaml",
            "Oligodendrocyte_mouse": "configs/noEmbedding/oligodendrocyte.yaml",
            #"Bone_marrow_mouse": "configs/noEmbedding/mouse_bonemarrow.yaml",
            #"Cerebral_cortex_human": "configs/noEmbedding/human_cerebral.yaml",
            #"Hematopoiesis_mouse": "configs/noEmbedding/mouse_hema.yaml",
            #"Spermatogenisis": "configs/noEmbedding/spermatogenesis.yaml",
            #"Old_brain": "configs/noEmbedding/old_brain.yaml",
            #"Prefrontal_cortex": "configs/noEmbedding/pfc.yaml",
            #"Intestinal_organoids":"configs/noEmbedding/GSE128365.yaml"
        }
    
    for name, path in datasets.items():
        print(f"Processing dataset: {name}")
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        adata = sc.read(config['path'])
        dataset_name = config['dataset_name']
        basis = config['basis']
        #neighbors = config['number_neighbors_to_sample_from']   

        Z_expr = f"X_{basis}"
        Z_velocity_vector = f"velocity_{basis}"
        cm = 1 / 2.54  # centimeters in inches
        figsize = (4.4 * cm, 3.3 * cm)

        ### Neighborhoods
        number_of_cells = [600, 800, 1200, 1600] # Run this script multiple timeswith different neighbors (100, 200, 300, 400)
        interesting_cells = [0,1,2,8,15,42]

        fig, axes = plt.subplots(
            len(number_of_cells), len(interesting_cells), 
            figsize=(figsize[0] * len(number_of_cells), figsize[1] * len(interesting_cells)),
            dpi=300, 
            constrained_layout=True
        )

        for i, neighbors in enumerate(number_of_cells):
            nn_indices = find_neighbors(adata.obsm[Z_expr], k_neighbors=neighbors)
            selected_neighbors = find_neighbors_in_direction_of_velocity(torch.tensor(adata.obsm[Z_expr]), torch.tensor(
                adata.obsm[Z_expr] + adata.obsm[Z_velocity_vector]), nn_indices)


            for j, cell in enumerate(interesting_cells):
                ax = axes[i,j]
                plot_neighborhood(cell, adata.obsm[Z_expr], nn_indices, ax=ax, s=0.3, vector_friendly=True,
                      selected_neighbors=selected_neighbors[cell])
                ax.set_aspect('equal')

                print("Done with", i+j)

        plt.savefig(
            f"results/neighbors/{dataset_name}-neighborhood_summary2.pdf",
            transparent=True, dpi=600)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        main()
    except RuntimeError:
        pass