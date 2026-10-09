
# Load required libraries
print("R script started. Loading libraries...")
library(argparse)
library(tidyverse)
library(anndataR)
library(dyngen)
library(reticulate)
library(parallelly)

# use_condaenv("scanpy")
use_condaenv("dyngen", required = TRUE)

options(timeout = 600)
options(download.file.method = "libcurl")

# Set up argument parser
parser <- ArgumentParser()
parser$add_argument("-n", "--num_cells", type = "integer", default = 100, help = "Number of cells to be simulated")
parser$add_argument("-b", "--backbone", type = "character", default = "linear",
    choices = c("linear", "bifurcating", "bifurcating_converging", "bifurcating_cycle", "bifurcating_loop",
    "binary_tree", "consecutive_bifurcating", "trifurcating", "converging", "cycle", "disconnected"), 
    help = "Dynamic process topology for dyngen")
parser$add_argument("--num_genes", type = "integer", default = 200, help = "Number of genes")
parser$add_argument("-o", "--out_dir", type = "character", default = "sim_data", help = "Output directory")
parser$add_argument("--seed", type = "integer", default = 42, help = "Random seed for reproducibility")
parser$add_argument("--total_time", type = "double", default = 350, help = "Total simulation time")
parser$add_argument("--time_points", type = "integer", default = 20, help = "Number of timepoints sampled")

args <- parser$parse_args()

# Create output directory if it doesn't exist
dir.create(args$out_dir, showWarnings = FALSE, recursive = TRUE)

# Set chosen random seed for reproducibility
set.seed(args$seed)

# Set up basic model
backbone <- switch(args$backbone,
  linear = backbone_linear(),
  bifurcating = backbone_bifurcating(),
  bifurcating_converging = backbone_bifurcating_converging(),
  bifurcating_cycle = backbone_bifurcating_cycle(),
  bifurcating_loop = backbone_bifurcating_loop(),
  binary_tree = backbone_binary_tree(num_modifications = 2),
  consecutive_bifurcating = backbone_consecutive_bifurcating(),
  trifurcating = backbone_trifurcating(),
  converging = backbone_converging(),
  cycle = backbone_cycle(),
  disconnected = backbone_disconnected()
)

num_cells <- args$num_cells
num_tfs <- nrow(backbone$module_info)
num_targets <- round((args$num_genes - num_tfs) / 2)
num_hks <- args$num_genes - num_targets - num_tfs
total_time <- args$total_time
time_points <- args$time_points

num_simulations <- min(100, max(10, num_cells / 10))
census_interval <- (total_time * num_simulations) / (20 * num_cells)

if (census_interval < 0.1) {
  census_interval <- 0.1
  num_simulations <- ceiling((20 * num_cells) / (total_time / census_interval))
}

print(paste("Using census_interval:", census_interval))
print(paste("Using num_simulations:", num_simulations))

out <-
  initialise_model(
    backbone = backbone,
    num_cells = num_cells,
    num_tfs = num_tfs,
    num_targets = num_targets,
    num_hks = num_hks,
    simulation_params = simulation_default(
      total_time = total_time,
      census_interval = census_interval,
      ssa_algorithm = ssa_etl(tau = 300 / 3600),
      experiment_params = simulation_type_wild_type(
        num_simulations = num_simulations
      ),
      compute_rna_velocity = TRUE
    ),
    experiment_params = experiment_synchronised(
      num_timepoints = time_points,
      pct_between = 0
    ),
    num_cores = 1
  ) %>%
  generate_dataset()

print("Simulation completed. Converting to anndata...")
adata <- wrap_dataset(out$model, format = "anndata")

py_adata <- reticulate::r_to_py(adata)

output_file <- file.path(args$out_dir,
  paste("dyngen_", args$backbone, "_", args$num_cells, "_cells_", args$num_genes, "_genes_", args$total_time, "_total_", args$time_points, "_points_seed_", args$seed, ".h5ad",sep = ""))

ad_py_module <- reticulate::import("anndata")
py_adata$write(output_file)

print("Saved simulation data")