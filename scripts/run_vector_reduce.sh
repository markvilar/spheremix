#!/usr/bin/bash

VECTOR_DIR="/data/exos_01/acfr_topological_mapping/acfr_image_vector_arrays"

# Invoke mixture fitting example script
uv run examples/reduce_vectors.py \
    --input "${VECTOR_DIR}/image_vectors_camera_bundle_1_mono.npy" \
    --output "./data/image_vectors_mono_d128_site1.npy" \
    --components 128

uv run examples/reduce_vectors.py \
    --input "${VECTOR_DIR}/image_vectors_camera_bundle_2_mono.npy" \
    --output "./data/image_vectors_mono_d128_site2.npy" \
    --components 128

uv run examples/reduce_vectors.py \
    --input "${VECTOR_DIR}/image_vectors_camera_bundle_3_mono.npy" \
    --output "./data/image_vectors_mono_d128_site3.npy" \
    --components 128

uv run examples/reduce_vectors.py \
    --input "${VECTOR_DIR}/image_vectors_camera_bundle_4_mono.npy" \
    --output "./data/image_vectors_mono_d128_site4.npy" \
    --components 128

uv run examples/reduce_vectors.py \
    --input "${VECTOR_DIR}/image_vectors_camera_bundle_5_mono.npy" \
    --output "./data/image_vectors_mono_d128_site5.npy" \
    --components 128

# ------------------------
# ---- Reduce to D256 ----
# ------------------------

# Invoke mixture fitting example script
uv run examples/reduce_vectors.py \
    --input "${VECTOR_DIR}/image_vectors_camera_bundle_1_mono.npy" \
    --output "./data/image_vectors_mono_d256_site1.npy" \
    --components 256

uv run examples/reduce_vectors.py \
    --input "${VECTOR_DIR}/image_vectors_camera_bundle_2_mono.npy" \
    --output "./data/image_vectors_mono_d256_site2.npy" \
    --components 256

uv run examples/reduce_vectors.py \
    --input "${VECTOR_DIR}/image_vectors_camera_bundle_3_mono.npy" \
    --output "./data/image_vectors_mono_d256_site3.npy" \
    --components 256

uv run examples/reduce_vectors.py \
    --input "${VECTOR_DIR}/image_vectors_camera_bundle_4_mono.npy" \
    --output "./data/image_vectors_mono_d256_site4.npy" \
    --components 256

uv run examples/reduce_vectors.py \
    --input "${VECTOR_DIR}/image_vectors_camera_bundle_5_mono.npy" \
    --output "./data/image_vectors_mono_d256_site5.npy" \
    --components 256
