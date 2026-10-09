import itertools
import math
import pickle

import anndata as ad
import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvelo
import seaborn as sns
import shap
import torch
import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import stats
from scipy.spatial import distance
from sklearn import ensemble, linear_model, preprocessing
from sklearn.inspection import permutation_importance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from tqdm import tqdm
from velotest.hypothesis_testing import run_hypothesis_test_on
from velotest.neighbors import find_neighbors, find_neighbors_in_direction_of_velocity



def add_non_nan_features(matrix, names, final_matrix, final_features):
    if sp.issparse(matrix):
        if np.isnan(matrix.data).any():
            matrix = matrix.toarray() # Convert to dense for row/col filtering
        else:
            final_matrix.append(matrix)
            final_features.extend(names)
            return final_matrix, final_features

    valid_cols_mask = ~np.isnan(matrix).any(axis=0)
    
    if valid_cols_mask.any(): # If at least one column is valid
        # Keep only valid columns
        final_matrix.append(matrix[:, valid_cols_mask])
        # Keep only names where the mask is True
        filtered_names = [name for i, name in enumerate(names) if valid_cols_mask[i]]
        final_features.extend(filtered_names)

    return final_matrix, final_features

def prepare_feature_matrix(adata_copy, cluster, basis, n_neighbors, use_cell_position = True, extra_features = False):
    ### Restack the anndataobject for learning
    X_spliced = adata_copy.layers['spliced']
    X_unspliced = adata_copy.layers['unspliced'] 
    # Velocity
    X_velocity_full = adata_copy.layers['velocity']
    valid_genes_mask = ~np.isnan(X_velocity_full).any(axis=0)
    # Splice/unpsliced totla and ratio
    total_spliced = np.asarray(X_spliced.sum(axis=1)).reshape(-1, 1)
    total_unspliced = np.asarray(X_unspliced.sum(axis=1)).reshape(-1, 1)
    #ratio_su = total_spliced / (total_unspliced + 1e-9).  # feautres was not taken into the final analysis

    # Other features saved in the object
    #le = preprocessing.LabelEncoder()
    #clusters = le.fit_transform(adata_copy.obs[cluster]).reshape(-1, 1) # feautres was not taken into the final analysis

    # Use the number of neighbors in the direction of velocity
    nn_indices = find_neighbors(torch.tensor(adata_copy.obsm[f'X_{basis}']), k_neighbors=n_neighbors) # Find all neighbors in the 2D embedding 
    neighbors = find_neighbors_in_direction_of_velocity(torch.tensor(adata_copy.obsm[f'X_{basis}']), torch.tensor(adata_copy.obsm[f'X_{basis}']+adata_copy.obsm[f'velocity_{basis}']), nn_indices, 22.5) # Find all neighbor in the direction of velocity in the 2D embedding
    num_neighbors = np.array([len(neigh) for neigh in neighbors]).reshape(-1, 1)

    # Compute different velocities
    highD_velocity = np.linalg.norm(adata_copy.layers['velocity'][:,valid_genes_mask], axis=1).reshape(-1, 1)
    #twoD_velocity = np.linalg.norm(adata_copy.obsm[f'velocity_{basis}'], axis=1).reshape(-1, 1) # feautres was not taken into the final analysis
    #confidence = adata_copy.obs['velocity_self_transition'].values.reshape(-1, 1)

    # Extract cell position
    if use_cell_position:
        position_mapping = {"edge": 0, "rim": 1, "inner": 2}
        cell_position = adata_copy.obs['rim'].map(position_mapping).astype(float).values.reshape(-1, 1)
        pos_feature_names = ["spatial_depth"]

    if extra_features:
        # Entropy
        V = X_velocity_full[:, valid_genes_mask]
        if sp.issparse(V):
            V = V.toarray()  
        V_abs = np.nan_to_num(np.abs(V), nan=0.0) + 1e-12
        velocity_entropy = stats.entropy(V_abs, axis=1).reshape(-1, 1)

        # Consistency Score (velocity coherence)
        velocity = adata_copy.layers["velocity"]
        nn_indices = find_neighbors(adata_copy.layers['Ms'], k_neighbors=n_neighbors)

        corr = []
        for cell in range(velocity.shape[0]):
            neighbors = nn_indices[cell]
            current_cell_v = velocity[cell].toarray().flatten() if hasattr(velocity, "toarray") else velocity[cell].flatten()
            neighbor_v = velocity[neighbors].toarray() if hasattr(velocity, "toarray") else velocity[neighbors]

            valid_genes = ~np.isnan(current_cell_v) & ~np.any(np.isnan(neighbor_v), axis=0)
            current_cell_v_clean = current_cell_v[valid_genes].reshape(1, -1)
            neighbor_v_clean = neighbor_v[:, valid_genes]
            
            cos_sim = cosine_similarity(current_cell_v_clean, neighbor_v_clean)
            corr.append(np.mean(cos_sim))
        corr = np.array(corr).reshape(-1, 1)
        
    # Build matrix and check everyhting for nan values before stacking
    matrices = []
    final_feature = []

    matrices, final_feature = add_non_nan_features(total_spliced, ["total_spliced"], matrices, final_feature)
    matrices, final_feature = add_non_nan_features(total_unspliced, ["total_unspliced"], matrices, final_feature)
    matrices, final_feature = add_non_nan_features(num_neighbors, ["num_neighbors"], matrices, final_feature)
    matrices, final_feature = add_non_nan_features(highD_velocity, ["highD_velocity"], matrices, final_feature)
    #matrices, final_feature = add_non_nan_features(confidence, ["confidence"], matrices, final_feature)

    if use_cell_position == True: 
        matrices, final_feature = add_non_nan_features(cell_position, pos_feature_names, matrices, final_feature)

    if extra_features:
        matrices, final_feature = add_non_nan_features(velocity_entropy, ["velocity_entropy"], matrices, final_feature)
        matrices, final_feature = add_non_nan_features(corr, ["consistency"], matrices, final_feature)

    X = np.hstack(matrices)
    features = final_feature

    return X, features

def prepare_classification(dis):
    sorted_values = np.sort(dis)
    middle_index = sorted_values.size // 2
    threshold = sorted_values[middle_index]

    # Create response for classification
    response = np.zeros(len(dis))
    response[dis > threshold] = 1 
    return response

def tune_rf_across_datasets(datasets, use_cell_position, use_extra):
    """
    Finds the best Random Forest hyperparameters for each dataset.
    """
    best_params_per_dataset = {}
    
    # Define a search space slightly around your current fixed parameters
    param_dist = {
        'n_estimators': [100, 500,],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None],
    }
    
    for name, path in datasets.items():
        print(f"-> Tuning for dataset: {name}")
        with open(path, "r") as f:
            config = yaml.safe_load(f)  
            
        # 1. Load and prepare data (same as your main loop)
        adata = sc.read(config['path'])
        basis = config['basis']
        cluster = config['cluster']
        n_neighbors = config['number_neighbors_to_sample_from']

        upper = adata.obsm['upper_max_neighbors']
        best = adata.uns['best_embedding']
        non_empty_indices = adata.uns['non_empty_indices']

        adata_copy = adata[non_empty_indices, :].copy()
        upper = upper[non_empty_indices]
        distances = np.subtract(upper, best)

        X, features = prepare_feature_matrix(adata_copy, 
                                             cluster=cluster, 
                                             basis=basis, 
                                             n_neighbors=n_neighbors,
                                             use_cell_position=use_cell_position,
                                             extra_features=use_extra)
        response = prepare_classification(distances)

        # Subset to the top/bottom 200 cells
        sorted_idx = np.argsort(distances)
        bottom_200_idx = sorted_idx[:200]
        top_200_idx = sorted_idx[-200:]

        extreme_indices = np.concatenate([bottom_200_idx, top_200_idx])
        X_extremes = X[extreme_indices, :]
        y_extremes = np.concatenate([np.zeros(200), np.ones(200)])
        # End of subsetting part
        
        # Run Randomized Search
        rf = ensemble.RandomForestClassifier(random_state=0)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 

        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=15, 
            cv=cv,
            scoring='accuracy', 
            n_jobs=-1,          
            random_state=42
        )

        # Adapt for which model you are optimizing
        search.fit(X_extremes, y_extremes)
        
        print(f"   Best params: {search.best_params_}")
        print(f"   Best score: {search.best_score_:.4f}\n")
        
        best_params_per_dataset[name] = search.best_params_
        
    return best_params_per_dataset

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

    tuning_results = tune_rf_across_datasets(datasets, use_cell_position=True, use_extra=False)
    df = pd.DataFrame.from_dict(tuning_results, orient='index')
    consensus = {}
    print("=== Hyperparameter Voting Across Datasets ===")
    
    for param in df.columns:
        # Get counts for each setting
        counts = df[param].value_counts(dropna=False)
        top_value = counts.index[0]
        top_count = counts.iloc[0]
        percentage = (top_count / len(df)) * 100
        
        consensus[param] = top_value
        print(f"\n• {param}:")
        print(f"   Consensus: {top_value} (won in {top_count}/{len(df)} datasets, {percentage:.0f}%)")
        print(f"   Distribution: {dict(counts)}")
        
    print("\n=== Ready-to-Use Dict ===")
    print(consensus)

if __name__ == "__main__":
    try:
        #mp.set_start_method('spawn', force=True)
        main()
    except RuntimeError:
        pass