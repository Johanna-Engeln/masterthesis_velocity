import multiprocessing as mp
import numpy as np
import argparse
import os
import scvelo
import scanpy as sc
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import torch

from velotest.hypothesis_testing import run_hypothesis_test_on, correct_for_multiple_testing
from velotest.explicit_hypothesis_testing import compute_p_values
from velotest.neighbors import find_neighbors_in_direction_of_velocity, find_neighbors
from velotest.plotting import arrow_plot, plot_uniformity_histogram, plot_best_possible_velocities_statistic, compute_angle_on_gridplot_between

def calculateBound(adata, k=None, neighbors=True, number_neighbors=200): 
    if neighbors is True:
        nn_indices_optimal = find_neighbors(torch.tensor(adata.layers['Ms']), k_neighbors=number_neighbors)

    maxs = torch.tensor([-2.]*len(adata))
    for i in range(len(adata)):
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

def bestVelocityEmbedded(adata, basis, test_neighbors = 200):
    uncorrected_p_values, h0_rejected, debug_dict = run_hypothesis_test_on(adata, basis=basis, number_neighbors_to_sample_from=test_neighbors, null_distribution="velocities-explicit", cosine_empty_neighborhood=None, correction="bonferroni", restrict_to_velocity_genes=True, exclusion_degree=10)
    statistics = debug_dict['statistics']
    non_empty_neighborhoods_indices = np.where(uncorrected_p_values != 2)[0]

    # Calculate emmbeding test optimal statistic
    best_value = [statistics.get_max_value(offset=True)[1] for statistics in statistics if statistics is not None]
    original_value = [statistics.get_values()[0] for statistics in statistics if statistics is not None]

    return best_value, original_value, non_empty_neighborhoods_indices

def determine_significant_vectors(gamma_rad, non_empty_neighborhoods_indices, statistics):
    uncorrected_p_values = compute_p_values(statistics, exclusion_rad=gamma_rad)
    pvals_corrected = correct_for_multiple_testing(uncorrected_p_values[uncorrected_p_values != 2], 'bonferroni')
    h0_rejected_temp = pvals_corrected < 0.05
    h0_rejected = np.zeros(len(statistics), dtype=bool)
    h0_rejected[non_empty_neighborhoods_indices] = h0_rejected_temp
    return h0_rejected, uncorrected_p_values

# Plot embedding with significant velocities highlighted
def plot_streamplot(adata, basis, output_path, dataset_name, color, cluster_colors, figsize=(5, 4)):
    # Start plot frame
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(dpi=450, figsize=(4.4 * cm, 3.5 * cm))
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
    plt.savefig(f"/mnt/volume/Johanna_Velotest/sim_data/velotest_2/{dataset_name}_{basis}_streamplot.pdf",
        transparent=True, dpi=450)
    plt.close()
    pass

def plot_significant_velocities(adata, basis, corrected_p_values, h0_rejected, color, dataset_name):
    # Define arguments for arrow_plot
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(dpi=450, figsize=(4.4 * cm, 3.5 * cm))
    Z_expr = f"X_{basis}"
    Z_velocity_vector = f"velocity_{basis}"

    arrow_plot(adata.obsm[Z_expr], adata.obsm[Z_expr] + adata.obsm[Z_velocity_vector], 
            corrected_p_values,
            h0_rejected,
            labels=adata.obs[color], 
            #label_colormap=dict(zip(adata.obs[color].cat.categories, adata.uns[cluster_colors])), 
            multiplier=0.15,
            plot_legend=False,
            ax=ax)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.text(0.975, 0.025, f'{np.sum(h0_rejected) / len(adata) * 100:.1f}%', transform=fig.transFigure,
                fontsize=5, ha="right")
    plt.savefig(f"/mnt/volume/Johanna_Velotest/sim_data/velotest_2/{dataset_name}_significant_velocities.pdf",
        transparent=True, dpi=450)
    return 

def plot_histogramm_p_values(statistics, non_empty_neighborhoods_indices, dataset_name):
    # Determine parameters
    cm = 1 / 2.54 

    h0_rejected_0, uncorrected_p_values_0 = determine_significant_vectors(None, non_empty_neighborhoods_indices,
                                                                            statistics)
    fig, ax = plt.subplots(dpi=450, constrained_layout=True,
                            figsize=((3.5) * cm, 1.95 * cm))  
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax.set_xticks([0, 1], labels=[0, 1], fontsize=4)
    ax.tick_params(axis='y', labelsize=4)

    plot_uniformity_histogram(uncorrected_p_values_0[uncorrected_p_values_0 < 2], ax=ax)
    assert plt.ylim()[1] <= 5.43
    plt.ylim(top=5.43)
    plt.xlabel("p-value", fontsize=4)
    plt.ylabel("Density", fontsize=4)
    plt.savefig(f"/mnt/volume/Johanna_Velotest/sim_data/velotest_2/{dataset_name}_histogramm_p_values.pdf",
        transparent=True, dpi=450)
    plt.show()

def plot_bounds(test_optimal, general_bound, bound_optimal_emb, dataset_name,
             ax=None, limits=None):
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)

    palette = sns.color_palette("colorblind", 5)
    ax.set_prop_cycle(color=palette)

    sns.kdeplot(test_optimal, clip=(-1, 1), fill=False, label="Test-optimal velocity on original visualization", linestyle="--")
    sns.kdeplot(general_bound, clip=(-1, 1), fill=False, label="General upper bound")  # max of all
    sns.kdeplot(bound_optimal_emb, clip=(-1, 1), fill=False,
                label="Upper bound assuming optimal \nneighborhood embedding")  # max of neighborhood

    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.legend()
    
    ax.set_xlim(-1,1)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=4))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=3))

    plt.savefig(f"/mnt/volume/Johanna_Velotest/sim_data/velotest_2/{dataset_name}_upperbounds.pdf", transparent=True)
    return

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--file_name", type=str, help="Path to the input file containing the AnnData object.")
    args.add_argument("--output_dir", type=str, default="/mnt/volume/Johanna_Velotest/sim_data/", help="Path to the output directory where results will be saved.")

    args = args.parse_args()

    file_name_parts = args.file_name.split("_")
    file_name_parts[0] = "dynamical"
    new_file_name = "_".join(file_name_parts)
    file_path = os.path.join(args.output_dir, args.file_name)
    adata = sc.read_h5ad(file_path)

    # Adapt naming to scVelo readable naming
    adata.layers['spliced'] = adata.layers['counts_spliced'].copy()
    adata.layers['unspliced'] = adata.layers['counts_unspliced'].copy()
    progressions = adata.uns['traj_progressions'].copy()
    progressions.set_index('cell_id', inplace=True)
    adata.obs = progressions

    # Compute velocities and plot
    scvelo.pp.filter_and_normalize(adata, n_top_genes=2000)
    scvelo.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scvelo.tl.recover_dynamics(adata, n_jobs=1)
    sc.tl.tsne(adata)
    basis = 'tsne'
    scvelo.tl.velocity(adata, mode='dynamical')
    scvelo.tl.velocity_graph(adata)
    #scvelo.pl.velocity_embedding(adata, basis=basis, color='from', title= args.file_name, show=False)
    #plt.savefig(f"/mnt/volume/Johanna_Velotest/sim_data/velotest/{new_file_name}_streamplot.pdf",
    #   transparent=True, dpi=450)
    scvelo.tl.velocity_embedding(adata, basis=basis)

    # Compute bounds
    neighbors = int(adata.n_obs * 0.05)
    general = calculateBound(adata, neighbors=False)
    upper_neigh = calculateBound(adata, neighbors=True, number_neighbors=neighbors)
    # Compute velotest results
    uncorrected_p_values, h0_rejected, debug_dict = run_hypothesis_test_on(adata, basis=basis, number_neighbors_to_sample_from=neighbors, null_distribution="velocities-explicit", cosine_empty_neighborhood=None, correction="bonferroni", restrict_to_velocity_genes=True, exclusion_degree=10)
    statistics = debug_dict['statistics']
    non_empty_neighborhoods_indices = np.where(uncorrected_p_values != 2)[0]

    # Calculate emmbeding test optimal statistic
    best = [statistics.get_max_value(offset=True)[1] for statistics in statistics if statistics is not None]
    original = [statistics.get_values()[0] for statistics in statistics if statistics is not None]

    # Save velotest results and velocity computation in anndata object
    adata.obsm['general_bound'] = general.numpy()
    adata.obsm['upper_max_neighbors'] = upper_neigh.numpy()
    adata.uns['non_empty_indices'] = non_empty_neighborhoods_indices
    adata.uns['best_embedding'] = best
    adata.uns['original_cosine_value'] = original
    #adata.write_h5ad(f"/mnt/volume/Johanna_Velotest/sim_data/{new_file_name}")

    # Plot simple velotest results
    plot_streamplot(adata, basis, "/mnt/volume/Johanna_Velotest/sim_data/velotest_2", dataset_name=new_file_name, color='from', cluster_colors='from_colors')
    plot_significant_velocities(adata, basis, uncorrected_p_values, h0_rejected, 'from', dataset_name=new_file_name)
    plot_histogramm_p_values(statistics, non_empty_neighborhoods_indices, dataset_name=new_file_name)
    plot_bounds(best, general.numpy(), upper_neigh.numpy(), dataset_name=new_file_name)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        main()
    except RuntimeError:
        pass