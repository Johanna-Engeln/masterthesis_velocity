import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo
import torch
import yaml
from scipy import stats
from tqdm import tqdm
import velotest
from velotest.hypothesis_testing import run_hypothesis_test_on
from velotest.neighbors import find_neighbors_in_direction_of_velocity, find_neighbors

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

    pca = {}

    for name, path in datasets.items():
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        adata = sc.read(config['path'])
        scvelo.tl.velocity_embedding(adata, basis='pca')
        neighbors = config['number_neighbors_to_sample_from']
        uncorrected_p_values, h0_rejected, debug_dict = run_hypothesis_test_on(adata, basis='pca', number_neighbors_to_sample_from=neighbors, null_distribution="velocities-explicit", cosine_empty_neighborhood=None, correction="bonferroni", restrict_to_velocity_genes=True, exclusion_degree=10)
        statistics = debug_dict['statistics']
        original_value = [statistics.get_values()[0] for statistics in statistics if statistics is not None]
        upper = adata.obsm['upper_max_neighbors']
        pca[name] = stats.wasserstein_distance(original_value, upper)

    df = pd.DataFrame([pca])
    print("Here")
    df.to_csv("results/distances/0908_Was_all_ori_pca.csv", index=False)
    print("Done")



if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        main()
    except RuntimeError:
        pass