#!/usr/bin/bash

VECTOR_DIR="/data/exos_01/acfr_topological_mapping/acfr_image_vector_arrays"

# Invoke mixture fitting example script
# uv run examples/run_mixture_em.py --file "./data/Abstracts256_X.npy"
uv run examples/run_mixture_em.py --file "./data/image_vectors_mono_d128_site1.npy"
