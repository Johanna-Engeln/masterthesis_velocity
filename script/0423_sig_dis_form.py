import multiprocessing as mp

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import velotest
import yaml
from scipy import stats
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from velotest.hypothesis_testing import run_hypothesis_test_on
from velotest.neighbors import find_neighbors, find_neighbors_in_direction_of_velocity



def main():
    datasets = {
            "Pancreas_dynamical": "/mnt/volume/Johanna_Velotest/configs/pancreas_dynamic.yaml",
            "Pancreas_stochastic": "/mnt/volume/Johanna_Velotest/configs/pancreas_stochastic.yaml",
            "Dentate_gyrus": "/mnt/volume/Johanna_Velotest/configs/dentategyrus.yaml",
            "Gastrulation_erythroid": "/mnt/volume/Johanna_Velotest/configs/gastrulation.yaml",
            "Bone_marrow": "/mnt/volume/Johanna_Velotest/configs/bonemarrow.yaml",
            "PBMC": "/mnt/volume/Johanna_Velotest/configs/pbmc.yaml",
            "Covid": "/mnt/volume/Johanna_Velotest/configs/covid.yaml",
            "Retinal_development": "/mnt/volume/Johanna_Velotest/configs/retinal_development.yaml",
            "Human_HSC": "/mnt/volume/Johanna_Velotest/configs/humanHSC.yaml",
            "RPE_1": "/mnt/volume/Johanna_Velotest/configs/noEmbedding/intestinal_organoids.yaml",
            "Hindbrain_mouse": "/mnt/volume/Johanna_Velotest/configs/noEmbedding/hindbrain.yaml",
            "Oligodendrocyte_mouse": "/mnt/volume/Johanna_Velotest/configs/noEmbedding/oligodendrocyte.yaml",
            "Bone_marrow_mouse": "/mnt/volume/Johanna_Velotest/configs/noEmbedding/mouse_bonemarrow.yaml",
            "Cerebral_cortex_human": "/mnt/volume/Johanna_Velotest/configs/noEmbedding/human_cerebral.yaml",
            "Hematopoiesis_mouse": "/mnt/volume/Johanna_Velotest/configs/noEmbedding/mouse_hema.yaml",
            "Spermatogenisis": "/mnt/volume/Johanna_Velotest/configs/noEmbedding/spermatogenesis.yaml",
            "Old_brain": "/mnt/volume/Johanna_Velotest/configs/noEmbedding/old_brain.yaml",
            "Prefrontal_cortex": "/mnt/volume/Johanna_Velotest/configs/noEmbedding/pfc.yaml",
            "Intestinal_organoids":"/mnt/volume/Johanna_Velotest/configs/noEmbedding/GSE128365.yaml",
            "Zebra_NMP": "/mnt/volume/Johanna_Velotest/configs/noEmbedding/zebra_NMP.yaml"
        }
    results = {}

    for name, path in datasets.items():
        # Calculte PCA and original valuesk
        with open(path, "r") as f:
                config = yaml.safe_load(f)

        adata = sc.read(config['path'])
        neighbors = config['number_neighbors_to_sample_from']
        basis = config['basis']
        general = adata.obsm['general_bound']
        upper_bound = adata.obsm['upper_max_neighbors']
        best_value = adata.uns['best_embedding']
        non_empty_neighborhoods_indices = adata.uns['non_empty_indices'] 

        uncorrected_p_values, h0_rejected, debug_dict = run_hypothesis_test_on(adata, 
                                                                            basis=basis,
                                                                            number_neighbors_to_sample_from=neighbors, 
                                                                            null_distribution="velocities-explicit", 
                                                                            cosine_empty_neighborhood=None, 
                                                                            correction="bonferroni", 
                                                                            restrict_to_velocity_genes=True, 
                                                                            exclusion_degree=10)
        
        # Get distance value(y)
        distance = stats.wasserstein_distance(upper_bound, best_value)
        mse = mean_squared_error(upper_bound[non_empty_neighborhoods_indices], best_value)

        # Get significance value(x)
        significance = np.sum(h0_rejected) / len(adata) * 100

        # Calculate the moments 
        var_general = np.nanvar(general)
        var_upper = np.nanvar(upper_bound)
        var_best = np.nanvar(best_value)
        skew_general = stats.skew(general, nan_policy='omit')
        skew_upper = stats.skew(upper_bound, nan_policy='omit')
        skew_best = stats.skew(best_value, nan_policy='omit')
        kurt_general = stats.kurtosis(general, nan_policy='omit')
        kurt_upper = stats.kurtosis(upper_bound, nan_policy='omit')
        kurt_best = stats.kurtosis(best_value, nan_policy='omit')

        results[name] = {
            'significance': significance,
            'wasserstein': distance,
            'mse': mse,
            'cellcount': adata.n_obs,
            'median_general': np.median(general),
            'median_upper': np.median(upper_bound),
            'median_best': np.median(best_value),  
            'var_general': var_general,
            'skew_general': skew_general,
            'kurt_general': kurt_general,
            'var_upper': var_upper,
            'skew_upper': skew_upper,
            'kurt_upper': kurt_upper,
            'var_best': var_best,
            'skew_best': skew_best,
            'kurt_best': kurt_best
        }

        df = pd.DataFrame(results)
        df.to_csv("/mnt/volume/Johanna_Velotest/results/distances/comparison_sig_was_mse.csv")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        main()
    except RuntimeError:
        pass