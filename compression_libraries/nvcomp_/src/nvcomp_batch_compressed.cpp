/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#include <cuda_runtime.h>

#include <iostream>

// SMASH LIBRARIES
#include <nvcomp_batch_compressed.hpp>

BatchDataCompressed::BatchDataCompressed(const size_t &slices)
    : slices_(slices), max_chunk_size_(0) {
  cudaMallocHost(&h_ptrs_compression_, sizeof(*h_ptrs_compression_) * slices);
  cudaMalloc(&d_ptrs_, sizeof(*d_ptrs_) * slices);
  cudaMalloc(&d_sizes_, sizeof(*d_sizes_) * slices);
  cudaMalloc(&d_size_, sizeof(*d_size_));
}

BatchDataCompressed::~BatchDataCompressed() {
  cudaFree(d_ptrs_);
  cudaFree(d_sizes_);
  cudaFree(d_size_);
  if (max_chunk_size_) {
    cudaError_t error{cudaSuccess};
    for (uint64_t i_chunk = 0; error == cudaSuccess && i_chunk < slices_;
         ++i_chunk) {
      error = cudaFree(h_ptrs_compression_[i_chunk]);
    }
  }
  cudaFreeHost(h_ptrs_compression_);
}
