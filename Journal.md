# Project Journal 

In this document an explanation of the used code is provided. 

## Code used for the master's thesis

*Data preprocessing:*  
To save computational time the velocity computation and embeddings were saved once.
- script/0331_dataPreparation.ipynb
- script/0414_dataPreparation.ipynb

The bounds were saved to their respective data as well to save time:
- script/0511_addData.ipynb

*Run velotest:* 
This code generated results of Chapter 4.1.  
Here only one script was used and adapted to the specific test (dataset and embedding method).
- script/0414_runVelotest

To figure out the number of neighbors to use, this script was used: 
- script/0420_testNeighbors.py

*Summary of the bound investigation:*
- script/0712_summary_bounds.ipynb

*Compute Wasserstein distance between the velocity and the neighborhood-based upper bound:*
This code generated results of Chapter 4.2.  
The scripts were adapted to fulfill the different computational tasks of testing different embedding methods with different hyperparameter settings. 

For the test-optimal velocity: 
- script/0421_distances.py
- script/0728_pca.py

For the visualized velocity: 
- script/0831_distance_Velocity_direction.py

For VeloViz - the code was adapted once for the test-optimal velocity and then changed to the visualized velocity:
- script/0429_veloviz.R
- script/0429_velo_distance_for_R.py

For the subsampling (Chapter 4.3.3):
- script/0616_downsampleDistance.py

For primary visualization of the results:
- script/0529_plotsEmbeddingDistance.ipynb

*Compute structure preservation metrics for embedding methods:*
This code generated results presented in Chapter 4.2.
- script/0728_embbedding_qc.ipynb
- script/0729_embedding_qc.py
- script/0802_velo_qc_help.py
- script/0802_veloviz_qc.R

*Investigate the hyperparameter k (neighborhood size):*
This code generated the results of Chapter 4.3.2.  
To investigate the problem, first the bound was tested with different settings of *k*:
- script/0421_BoundNeighborSize.ipynb

This is the main notebook with visualizations:
- script/0427_differentK_influence.ipynb
- script/0911_helper_k.py

*Feature discovery with RF:*
This code was used to generate the results of Chapter 4.3.1
- script/0804_discoverFeatures.ipynb
- script/0911_final_classification.ipynb
- script/0916_RF_tuning.py

*Use the cosine similarity distribution:*
This code was used to generate the results presented in Chapter 4.3.3 and to find a good way to present Chapter 3.8.  
Note that in those notebooks also plots and computations that were not presented in the thesis are included.
- script/0617_proof.ipynb
- script/0707_incompleteBeta.ipynb


## Code not used in the master's thesis

Further in this repository there are scripts stored, that weren't used in the final thesis. 

We tried to figure out the first, second, third, and fourth order moment. 
- script/0423_sig_dis_form.py

We tried to figure out if there is a correlation between the significance value from velotest and the Wasserstein distance that was calculated. 

- script/0423_sig_vs_distances.ipynb

Further different approaches in the case of subsampling were used.
- script/0519_compute_k_subsets.py
- script/0730_downsampleDistance_clutser.py

Another approach was to investigate experimental factors with synthetic data. Therefore these files were used: 
- script/0602_runSimulation.sh
- script/0602_simulate_data.R
- script/0602_velocity.py
- script/0608_visulaizationDyngen.ipynb
- script/0712_runVelotest_file.py


## Personal notes
Command to use to attach volume to server again: "sudo mount /dev/vdb /mnt/volume/"