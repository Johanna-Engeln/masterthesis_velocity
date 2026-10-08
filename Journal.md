# Project Journal 

In this document and explanation of the used code is provided. 

## Code used for the master's thesis

*Data preprocessing:*  
To save computational time the velocity computation and embeddings were saved once.
- script/0331_dataPreparation.ipynb
- script/0414_dataPreparation.ipynb

The differnt bounds were save to its respective data as well
- script/0511_addData.ipynb

*Run velotest:* 
This code generated results of Chapter 4.1
Here only one script was used and adpated to the specific test (dataset and embedding method)
- script/0414_runVelotest

*Compute Wasserstein distance between the velocity and the neighborhood-based upper bound:*
This code generated results of Chapter 4.1
The scirpts were adapted to fullfil the different computational tasks of testing different embedding methods with different hyperparameter settings. 

For the test-optimal velocity: 
- script/0421_distances.py
- script/0728_pca.py

For the visualized velocity: 
- script/0831_distance_Velocity_direction.py

For VeloViz - the code was adated one for the test-optimal velocity and then change to the visualized velocity:
- script/0429_veloviz.R
- script/0429_velo_distance_for_R.py

For primalry visualization:
- script/0529_plotsEmbeddingDistance.ipynb

*Investigate the hyperparameter k (neighborhood size):*
This code generated the results of Chapter 4.3.2

To investigate the problem, first the bound was test with differen settings of *k*
- script/0421_BoundNeighborSize.ipynb


## Code not used in the master's thesis



## Personal notes:
Command to use to attach volume to server again: "sudo mount /dev/vdb /mnt/volume/"

## Week 27
- Tried to work with the von-Mises distribution as velocity estimate

## Week 26
- Worked on the mathematical proof to understand some mathematical connections
- Repeated to run subsampling to 2000 cells and than compute W-Dis of the different datasets
- Recreate the theoretical sampling with a real dataset. 

## Week 25
- Started to examine the hypothesis that "More cells in a dataset = higher W-Dis"
- Started to run subsampling to 2000 cells and than compute W-Dis of the different datasets (had a coding error...)

## Week 24 (break)
- Gave Manfred the Update

## Week 23
- Explored dyngen and tried to get influences of experimental time and sampling recreated to examine it as one determinate that influences the embedding
- Finished bigger update for Manfred and Sebastian

## Week 22
- Finalize some small task that were pilling up as leftovers
- Start writing thesis k part
- Prepared summary for Seabstian, k part, embedding part, Entropy and consitency score, model comparison, sythetic data

## Week 21
- Use subsamplig for k neighbors to 'proof' trend
- Play around with entropy and consistency score
- Download and try synthetic datasets

## Week 20
- Meeting mit Manfred und Sebastian
- Save bounds in anndata object
- Tried to find formular explaniation for the linearity between self-transition and upper bound (check code an paper)
- Finally started veloviz run
- finished k computations

## Week 19
- Create the github repository and start proper documentation
- 5 of 19 datasets had problems running veloviz script, try to fix those
- Run Linear regression with self transition - results are super confusing, asked Sebastian

## What was done sofar: 
- Figured out the neighbor parameter k of velotest 
- Split work in three parts (embedding, within datasets, between datasets)