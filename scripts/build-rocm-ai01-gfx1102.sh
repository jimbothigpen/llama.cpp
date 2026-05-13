#!/usr/bin/env bash
# Yggdrasil ROCm build for ai01 — NATIVE gfx1103 single-target (one-off experiment).
# Adapted from conventions/build-and-install.md ai01 ROCm recipe.

set -euo pipefail

cd /mnt/cephfs/0/Container/systems/ai00/users/builduser/llama-yggdrasil

rm -rf build-rocm-ai01
mkdir build-rocm-ai01
cd build-rocm-ai01

cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/llama-yggdrasil-rocm \
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="gfx1102" \
    -DGGML_AVX512=ON \
    -DGGML_AVX512_VBMI=ON \
    -DGGML_AVX512_VNNI=ON \
    -DGGML_AVX512_BF16=ON \
    -DCMAKE_C_FLAGS="-march=native -O3" \
    -DCMAKE_CXX_FLAGS="-march=native -O3"

ninja -j16

sudo rm -rf /opt/llama-yggdrasil-rocm/
sudo ninja install

echo
echo "=== BUILD + INSTALL COMPLETE ==="
ls /opt/llama-yggdrasil-rocm/bin/ | head -20
