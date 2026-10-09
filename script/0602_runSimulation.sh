#!/bin/bash

# Static parameters
OUT_DIR="/mnt/volume/Johanna_Velotest/sim_data"
BACKBONE="trifurcating"
NUM_CELLS=(2000 5000 10000 30000)
UM_GENES=2000
SEEDS=(1 2 5 12 42)
TOTAL_TIME=(100 350 1000 5000)
TIME_POINTS=(3 20 80 200)
echo "Run simulation script with the following parameters:"

for current_time in "${TOTAL_TIME[@]}"; do
  for points in "${TIME_POINTS[@]}"; do
    for current_cells in "${NUM_CELLS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        echo "Running simulation with seed ${SEED}..."
        echo "========================================="
        echo "Running: Time=${current_time}, Points=${points}, Cells=${current_cells}, Seed=${SEED}"
        echo "========================================="
        Rscript /mnt/volume/Johanna_Velotest/script/0602_simulate_data.R \
          --num_cells "${current_cells}" \
          --backbone "${BACKBONE}" \
          --num_genes "${NUM_GENES}" \
          --seed "${SEED}" \
          --total_time "${current_time}" \
          --time_points "${points}" 

        FILE_NAME="dyngen_trifurcating_${current_cells}_cells_${NUM_GENES}_genes_${current_time}_total_${points}_points_seed_${SEED}.h5ad"

        echo "Run scvelo"  

        python /mnt/volume/Johanna_Velotest/script/0602_velocity.py \
          --file_name "${FILE_NAME}" \
          --output_dir "${OUT_DIR}"
      done
    done
  done
done
