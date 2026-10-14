# Study of determinants of faithful RNA velocity embeddings

**Goal:** Evaluate faithful RNA velocity embeddings

**Author:** Johanna Engeln

Huge acknowledgement for this work go to my supervisors Sebastian Bischoff and Prof. Dr. Manfred Claassen

### Abstract

RNA velocity infers future cellular states from snapshot scRNA-seq data, enabling the study of cellular dynamics and trajectory development.
These computed velocities are typically projected onto two-dimensional embeddings for visual interpretation.
However, such low-dimensional spaces cannot fully capture the complexity of high-dimensional gene-expression data.
While the newly published tool velotest can evaluate the accuracy of an RNA velocity representation, it is still unknown which determinants influence a faithful RNA velocity embedding. 
In this work, velotest is applied on multiple datasets and different embedding methods to investigate the underlying factors for a reliable RNA velocity representation. 
These evaluations are based on a theoretical upper bound derived for each dataset, allowing a comparison of different embedding methods to determine which embedding preserves the true RNA velocity the best. 
This work provides an extensive usage of velotest, a deterministic comparison of different dimensionality reduction methods and an investigation into the determinants that influence a faithful RNA velocity representation.
By using the test-optimal velocity visualization introduced by velotest, it was found that t-SNE embeddings capture the velocities most faithfully regarding the theoretical upper bound.
Furthermore, an extended analysis indicates that a cell's position within the embedding impacts its velocity projection.

### How to run the code 

To recreate the results shown in the master thesis use the provided `velotest.yml` or `veloviz_env.yml`, to recreate the conda enviroment used. 
The data was downloaded separatly from public repositories in a folder named data and was processed with the provided scripts. To download the data read the material section of the thesis or use the links provided in the Data-Preparation Notebooks. 
The `Journal.md` provides a list with all scripts and what they were used for. 
