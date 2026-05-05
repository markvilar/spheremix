#!/usr/bin/bash

VECTOR_DIR="/data/exos_01/acfr_topological_mapping/acfr_image_vector_arrays"

# Invoke mixture fitting example script
uv run examples/fit_spherical_cauchy_mixture.py \
  --n-components 20 \
  --n-samples 200 \
  --n-features 100 \
  --rho 0.70 \
  --verbose
