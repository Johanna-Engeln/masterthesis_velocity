import scvelo
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from velotest.hypothesis_testing import run_hypothesis_test_on
from velotest.neighbors import find_neighbors

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
    uncorrected_p_values, h0_rejected, debug_dict = run_hypothesis_test_on(adata, basis=basis, number_neighbors_to_sample_from=number_neighbors, null_distribution="velocities-explicit", cosine_empty_neighborhood=None, correction="bonferroni", restrict_to_velocity_genes=True, exclusion_degree=10, n_workers=18)
    statistics = debug_dict['statistics']

    # Calculate emmbeding test optimal statistic
    best_value = [statistics.get_max_value(offset=True)[1] for statistics in statistics if statistics is not None]
    non_empty_neighborhoods_indices = np.where(uncorrected_p_values != 2)[0]

    return best_value, non_empty_neighborhoods_indices

#### MAIN function to calculate distance
def distance_calculation(adata, upper, neighbors):
    scvelo.tl.velocity_embedding(adata, basis='veloviz')
    print("In python calculating distance")
    best, non_empty_neighborhoods_indices = bestVelocityEmbedded(adata, basis="veloviz", number_neighbors=neighbors)
    distance = stats.wasserstein_distance(best, upper)
    print(distance)

    return distance