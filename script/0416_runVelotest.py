import multiprocessing as mp

import yaml
import scanpy as sc
import scvelo
import torch
import numpy as np
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from velotest.hypothesis_testing import run_hypothesis_test_on, correct_for_multiple_testing
from velotest.explicit_hypothesis_testing import compute_p_values
from velotest.neighbors import find_neighbors, find_neighbors_in_direction_of_velocity
from velotest.plotting import arrow_plot, plot_uniformity_histogram, plot_best_possible_velocities_statistic, compute_angle_on_gridplot_between


### Functions copied from script make_all_figures
def determine_significant_vectors(gamma_rad, non_empty_neighborhoods_indices, statistics):
    uncorrected_p_values = compute_p_values(statistics, exclusion_rad=gamma_rad)
    pvals_corrected = correct_for_multiple_testing(uncorrected_p_values[uncorrected_p_values != 2], 'bonferroni')
    h0_rejected_temp = pvals_corrected < 0.05
    h0_rejected = np.zeros(len(statistics), dtype=bool)
    h0_rejected[non_empty_neighborhoods_indices] = h0_rejected_temp
    return h0_rejected, uncorrected_p_values

# Plot streamplot of velocities on embedding
def plot_streamplot(adata, basis, output_path, dataset_name, color, cluster_colors, figsize=(5, 4)):
    # Start plot frame
    fig, ax = plt.subplots(dpi=450, figsize=figsize)
    ax.set(xticks=[], yticks=[])
    color_dict = dict(zip(adata.obs[color].cat.categories, adata.uns[cluster_colors]))

    # Plot original streamplot
    pl = scvelo.pl.velocity_embedding_stream(adata, basis=basis, 
                                             color=color, palette=color_dict,
                                             title='', frameon=False,
                                             ax=ax,  xlabel='', ylabel='', 
                                             legend_fontsize=5, legend_fontweight='normal', size=3,
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
    
# Plot embedding with significant velocities highlighted
def plot_significant_velocities(adata, basis, corrected_p_values, h0_rejected, output_path, dataset_name, color, cluster_colors, figsize=(4.4, 4)):
    # Start plot frame
    fig, ax = plt.subplots(dpi=450, figsize=figsize)
    Z_expr = f"X_{basis}"
    Z_velocity_vector = f"velocity_{basis}"

    # Plot significant velocities
    arrow_plot(adata.obsm[Z_expr], adata.obsm[Z_expr] + adata.obsm[Z_velocity_vector], 
            corrected_p_values,
            h0_rejected,
            labels=adata.obs[color], 
            label_colormap=dict(zip(adata.obs[color].cat.categories, adata.uns[cluster_colors])), 
            multiplier=0.15,
            plot_legend=False,
            ax=ax)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.text(0.975, 0.025, f'{np.sum(h0_rejected) / len(adata) * 100:.1f}%', transform=fig.transFigure, fontsize=7, ha="right")

    # Save plot
    plt.savefig(f"{output_path}/{dataset_name}_{basis}_significant_velocities.pdf",
        transparent=True, dpi=450)
    pass

def plot_histogramm_p_values(adata, basis, statistics, non_empty_neighborhoods_indices, output_path, dataset_name, figsize):
    # Calculate significant velocities
    h0_rejected_0, uncorrected_p_values_0 = determine_significant_vectors(None, non_empty_neighborhoods_indices, statistics)
    # Start plot frame 
    fig, ax = plt.subplots(dpi=450, constrained_layout=True, figsize=figsize)  

    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax.set_xticks([0, 1], labels=[0, 1], fontsize=6)
    ax.tick_params(axis='both', width=0.5, labelsize=6)

    plot_uniformity_histogram(uncorrected_p_values_0[uncorrected_p_values_0 < 2], ax=ax)
    #assert plt.ylim()[1] <= 5.43
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(top=5.5)
    ax.set_yticks([0, 5], labels=[0, 5], fontsize=6)
    plt.xlabel("p-value", fontsize=7, labelpad=0.5)
    plt.ylabel("Density", fontsize=7)
    plt.savefig(f"{output_path}/{dataset_name}_{basis}_histogramm_p_values.pdf",
        transparent=True, dpi=450)
    pass

def estimate_test_optimal_velocity(adata, basis, statistics, non_empty_neighborhoods_indices, color, cluster_colors):
    # Determine parameters
    Z_expr = f"X_{basis}"
    Z_velocity_vector = f"velocity_{basis}"

    best_possible_velocities = [statistic.get_max_value(offset=True)[0] for statistic in statistics if
                                    statistic is not None]
    best_velocity_range_mean = []
    for range_best in best_possible_velocities:
        best_velocity_range_mean.append((range_best[0] + range_best[1]) / 2)
    best_velocity_range_mean = np.array(best_velocity_range_mean)
    best_velocities = np.stack([np.cos(best_velocity_range_mean), np.sin(best_velocity_range_mean)], axis=-1)
    adata_best = adata.copy()
    adata_best.obsm[Z_velocity_vector] = np.zeros((adata.shape[0], 2))
    adata_best.obsm[Z_velocity_vector][non_empty_neighborhoods_indices] = best_velocities

    angles = compute_angle_on_gridplot_between(adata, adata_best, basis=basis)

    norm = Normalize(vmin=0, vmax=90)
    cmap = LinearSegmentedColormap.from_list("lightgrey_to_black", ["#eeefff", "#000000"])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    colors = sm.to_rgba(angles)

    adata_plotting = ad.AnnData(adata_best.X, obs=adata_best.obs.copy())
    adata_plotting.obsm[Z_expr] = adata_best.obsm[Z_expr]
    adata_plotting.obsm[Z_velocity_vector] = adata_best.obsm[Z_velocity_vector]
    adata_plotting.layers['velocity'] = adata_best.layers['velocity']
    adata_plotting.uns[cluster_colors] = adata_best.uns[cluster_colors]
    color_dict_plotting = dict(zip(adata_plotting.obs[color].cat.categories,
                                    adata_plotting.uns[cluster_colors]))
    return adata_plotting, colors, color_dict_plotting

def plot_test_optimal_velocity(adata_plotting, basis, color_dict_plotting, colors, output_path, dataset_name, color, titel='', figsize=(4.4, 3.5)):
    # Start plot frame
    fig, ax = plt.subplots(dpi=450, figsize=figsize)
    ax.set(xticks=[], yticks=[])

    # Plot test-optimal velocity (Changed arrows in darker black)
    pl = scvelo.pl.velocity_embedding_grid(adata_plotting, 
                                        basis=basis, 
                                        dpi=450, title=titel, 
                                        ax=ax, frameon=False, xlabel='', ylabel='', 
                                        fontsize=16, show=False, size=3,
                                        linewidth=.25, arrow_size=2, arrow_length=2, 
                                        arrow_color=colors, edgecolors=None,
                                        rasterized=True, figsize=figsize, 
                                        color=color, 
                                        palette=color_dict_plotting
                                        )
    ax.set_aspect('equal')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    pl.set_rasterized(True)
    # Save plot
    plt.savefig(f"{output_path}/{dataset_name}_{basis}_test_optimal_velocity_direction.pdf",
        transparent=True, dpi=450)
    pass

def plot_best_possible_velocity(adata, basis, statistics, non_empty_neighborhoods_indices, output_path, dataset_name, figsize=(4.4, 4)):

    best_possible_velocities_statistic = [statistic.get_max_value()[1] for statistic in statistics if
                                          statistic is not None]
    fig, ax = plt.subplots(dpi=450, constrained_layout=True, figsize=figsize)
    plot_best_possible_velocities_statistic(adata.obsm[f'X_{basis}'], best_possible_velocities_statistic,
                                            non_empty_neighborhoods_indices,
                                            ax=ax,
                                            vector_friendly=True,
                                            max_value=0.97,
                                            cbar=False)
    ax.set_aspect('equal')
    # Save plot
    plt.savefig(f"{output_path}/{dataset_name}_{basis}_best_possible_velocities_statistic.pdf",
        transparent=True, dpi=450)
    return best_possible_velocities_statistic

def plot_histogram_best_possible_velocities(best_possible_velocities_statistic, output_path, dataset_name, basis):
    # Start plot frame
    cm = 1 / 2.54 
    figsize=(3.2 * cm, 1.8 * cm)
    fig, ax = plt.subplots(dpi=450, constrained_layout=True, figsize=figsize)

    plt.hist(best_possible_velocities_statistic, bins=100, density=True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    ax.set_xticks([-1, 0, 1], labels=[-1, "", 1], fontsize=4)
    ax.tick_params(axis='both', width=0.5, labelsize=4)
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlim(-1.07, 1.07)
    
    plt.ylim(top=8.5)
    plt.xlabel("Cosine similarity", fontsize=4, labelpad=0.5)
    plt.ylabel("Count", fontsize=4)
    plt.savefig(f"{output_path}/{dataset_name}_{basis}_histogramm_best_test_statistic.pdf", transparent=True, dpi=450)
    pass
    
# Run main part of the script
def main():
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
        #"Intestinal_organoids":"configs/noEmbedding/GSE128365.yaml",
        #"RPE_1": "configs/noEmbedding/intestinal_organoids.yaml",
        #"Hindbrain_mouse": "configs/noEmbedding/hindbrain.yaml",
        #"Oligodendrocyte_mouse": "configs/noEmbedding/oligodendrocyte.yaml",
        #"Bone_marrow_mouse": "configs/noEmbedding/mouse_bonemarrow.yaml",
        #"Cerebral_cortex_human": "configs/noEmbedding/human_cerebral.yaml",
        #"Hematopoiesis_mouse": "configs/noEmbedding/mouse_hema.yaml",
        "Spermatogenisis": "configs/noEmbedding/spermatogenesis.yaml",
        #"Old_brain": "configs/noEmbedding/old_brain.yaml",
        #"Prefrontal_cortex": "configs/noEmbedding/pfc.yaml",
        #"Zebra_NMP": "configs/noEmbedding/zebra_NMP.yaml"
    }

    # Set figure parameters
    cm = 1 / 2.54 
    figsize_pic = (3 * cm, 2.4 * cm)
    figsize_hist = (3 * cm, 1.7 * cm)
    output_path = "/mnt/volume/Johanna_Velotest/results/velotest"
 
    for name, path in datasets.items():
        print(f"Processing dataset: {name}")
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        adata = sc.read(config['path'])
        dataset_name = config['dataset_name']
        basis = config['basis']
        neighbors = config['number_neighbors_to_sample_from']   
        cluster = config['cluster']
        cluster_colors = config['cluster_colors']

        #adata = scvelo_processing(adata, basis=basis, mode=mode)
        plot_streamplot(adata, basis, output_path, dataset_name, cluster, cluster_colors, figsize=figsize_pic)

        # Run velotest
        uncorrected_p_values, h0_rejected, debug_dict = run_hypothesis_test_on(adata,
                                                                    number_neighbors_to_sample_from=neighbors,
                                                                    null_distribution="velocities-explicit",
                                                                    cosine_empty_neighborhood=None,
                                                                    correction="bonferroni",
                                                                    restrict_to_velocity_genes=True,
                                                                    exclusion_degree=10,
                                                                    n_workers=18,
                                                                    basis=basis)

        statistics = debug_dict['statistics']
        # This line needed to be added, because some cells still had a statistice but in reaged to the p_value they were empty neighborhoods. 
        for i, p_val in enumerate(uncorrected_p_values):
            if p_val == 2:
                statistics[i] = None
        non_empty_neighborhoods_indices = np.where(np.array(uncorrected_p_values) != 2)[0].astype(int)

        # Plot embedding with significant velocities highlighted
        plot_significant_velocities(adata, basis, uncorrected_p_values, h0_rejected, output_path, dataset_name, cluster, cluster_colors, figsize=figsize_pic)
        plot_histogramm_p_values(adata, basis, statistics, non_empty_neighborhoods_indices, output_path, dataset_name, figsize=figsize_hist)

        # Cumpute and plot test-optimal velocity
        adata_plotting, colors_plotting, color_dict_plotting = estimate_test_optimal_velocity(adata, basis, statistics, non_empty_neighborhoods_indices, cluster, cluster_colors)
        plot_test_optimal_velocity(adata_plotting, basis, color_dict_plotting, colors_plotting, output_path, dataset_name, cluster, figsize=figsize_pic)
        best_possible_velocities_statistic = plot_best_possible_velocity(adata, basis, statistics, non_empty_neighborhoods_indices, output_path, dataset_name, figsize=figsize_pic)
        plot_histogram_best_possible_velocities(best_possible_velocities_statistic, output_path, dataset_name, basis)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        main()
    except RuntimeError:
        pass