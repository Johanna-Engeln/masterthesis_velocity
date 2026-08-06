# Load libraries
library(veloviz)
library(reticulate)
library(yaml)
library(stringr)
print("R libraries loaded")
use_condaenv("/mnt/volume/Johanna_Velotest/enviroments/veloviz", required = TRUE)
sc <- import("scanpy")
scvelo <- import("scvelo")
source_python("/mnt/volume/Johanna_Velotest/script/0802_velo_qc_help.py")
print("Python libraries loaded")

# Load files 
files <- c(#"/mnt/volume/Johanna_Velotest/configs/pancreas_dynamic.yaml",
           # "/mnt/volume/Johanna_Velotest/configs/pancreas_stochastic.yaml",
            #"/mnt/volume/Johanna_Velotest/configs/dentategyrus.yaml",
            #"/mnt/volume/Johanna_Velotest/configs/gastrulation.yaml", # fail because of k - FIXED with the TRUE/FALSE swap
            #"/mnt/volume/Johanna_Velotest/configs/bonemarrow.yaml",
            "/mnt/volume/Johanna_Velotest/configs/pbmc.yaml", # fail Error in if (sum(curr < 0) > 0) { : missing value where TRUE/FALSE needed - FIXED
            "/mnt/volume/Johanna_Velotest/configs/covid.yaml",
            "/mnt/volume/Johanna_Velotest/configs/retinal_development.yaml",
            "/mnt/volume/Johanna_Velotest/configs/humanHSC.yaml", # curr contains negative values, cannot log normalize for PCA - FIXED
            "/mnt/volume/Johanna_Velotest/configs/noEmbedding/intestinal_organoids.yaml", # fail because of k - FIXED with the TRUE/FALSE swap
            "/mnt/volume/Johanna_Velotest/configs/noEmbedding/hindbrain.yaml",
            "/mnt/volume/Johanna_Velotest/configs/noEmbedding/oligodendrocyte.yaml",
            "/mnt/volume/Johanna_Velotest/configs/noEmbedding/mouse_bonemarrow.yaml",
            "/mnt/volume/Johanna_Velotest/configs/noEmbedding/human_cerebral.yaml",
            "/mnt/volume/Johanna_Velotest/configs/noEmbedding/mouse_hema.yaml",
            "/mnt/volume/Johanna_Velotest/configs/noEmbedding/spermatogenesis.yaml",
            "/mnt/volume/Johanna_Velotest/configs/noEmbedding/old_brain.yaml",
            "/mnt/volume/Johanna_Velotest/configs/noEmbedding/pfc.yaml",
            "/mnt/volume/Johanna_Velotest/configs/noEmbedding/GSE128365.yaml", # Same Error like humanHSC
            "/mnt/volume/Johanna_Velotest/configs/noEmbedding/zebra_NMP.yaml"
            )

# Initiate list to store stuff
all_results <- list()
row_index <- 1

for (path in files){
  # Load data
  yaml <- read_yaml(path)
  data_name <- yaml$dataset_name
  adata <- sc$read(yaml$path)
  genes <- adata$var_names$values
  cells <- adata$obs_names$values
  cluster_name <- yaml$cluster
  clusters <- adata$obs[[cluster_name]]
  names(clusters) <- cells
  cell.cols <- rainbow(8)[as.numeric(clusters)]
  names(cell.cols) <- names(clusters)

  # get velocity according to: https://jef.works/veloviz/scVeloVignette
  vel <- adata$layers[['velocity']]
  colnames(vel) <- genes
  rownames(vel) <- cells

  # get current
  curr <- adata$X
  colnames(curr) <- genes
  rownames(curr) <- cells

  # Handle NAs and negative values
  vel[is.na(vel)] <- 0
  curr[is.na(curr)] <- 0
  curr[curr < 0] <- 0

  vel.genes <- genes[colSums(vel, na.rm=T)>0]
  vel <- vel[,vel.genes]

  curr <- as.matrix(curr)[,vel.genes]
  curr <- t(curr)

  # Add filter steps due to the errors erousing in some datasets
  valid_cells <- colSums(curr) > 0
  curr <- curr[, valid_cells]
  vel <- vel[valid_cells, ]

  # compute projected
  proj <- curr + t(vel)
  proj[proj<0] <- 0

  print("Pre-computation done")

  # Get upper bound from anndata object
  upper <- adata$obsm[['upper_max_neighbors']]
  print(paste("Upper Bound: ", data_name))

  results = c()

  for (seed in 1:25){
    print(paste("Seed:", seed, " Set:", data_name))
    # build veloviz
    set.seed(seed)
    veloviz = buildVeloviz(
      curr = curr, proj = proj,
      normalize.depth = TRUE,
      use.ods.genes = FALSE,
      alpha = 0.05,
      pca = TRUE,
      nPCs = 20,
      center = TRUE,
      scale = TRUE,
      k = 30, # sample as UMAP default values here
      similarity.threshold = 0, # the following 3 paramters are chosen based on Veloviz recommondation for a rodust graph for unknown data
      distance.weight = 1,
      distance.threshold = 1,
      weighted = TRUE,
      #seed = 0,
      verbose = FALSE
    )

    print("Veloviz graph computed")
    emb.veloviz <- veloviz$fdg_coords

    valid_cells <- rownames(emb.veloviz)
    print(paste("Number of valid cells:", length(valid_cells)))

    adata_subset <- adata[valid_cells, ]$copy()
    adata_subset$obsm$`__setitem__`("X_veloviz", as.matrix(emb.veloviz))
    adata_subset$write_h5ad(paste0("data/veloviz/",data_name,"_veloviz.h5ad"))

    print("Check dimensions before python calculations")
    print(paste("curr dimensions (rows x cols):", nrow(curr), "x", ncol(curr)))
    print(paste("emb.veloviz dimensions (rows x cols):", nrow(emb.veloviz), "x", ncol(emb.veloviz)))
    print(paste("clusters length:", length(clusters)))

    print("Emnedding QC")
    curr_subset <- curr[, valid_cells, drop = FALSE]
    clusters_subset <- clusters[valid_cells]
    curr_python <- t(curr_subset)
    print(paste("curr transposed for Python (rows x cols):", nrow(curr_python), "x", ncol(curr_python)))

    trustworthiness = calculate_trustworthiness(curr_python, emb.veloviz)
    continuity = calculate_continuity(curr_python, emb.veloviz)
    knn = calculate_knn(curr_python, emb.veloviz)
    knc = calculate_knc(curr_python, emb.veloviz, clusters_subset)
    cpd = calculate_spearman(curr_python, emb.veloviz)
    print("knn, knc, cpd done")
    
    all_results[[row_index]] <- data.frame(
      dataset = data_name,
      seed = seed,
      embedding = "veloviz",
      trustworthiness = trustworthiness,
      continuity = continuity,
      knn = knn,
      cpd = cpd,
      knc = knc
    )
    
    row_index <- row_index + 1

  }
  df <- do.call(rbind, all_results)
  write.csv(df, "/mnt/volume/Johanna_Velotest/results/embbeding_qc/0803_embedding_veloviz.csv", quote = FALSE, row.names = FALSE)
}
df <- do.call(rbind, all_results)
write.csv(df, "/mnt/volume/Johanna_Velotest/results/embbedding_qc/0803_embedding_veloviz.csv", quote = FALSE, row.names = FALSE)