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
    : slices_(slices), compressing_(false) {
  h_ptrs_ = new char *[slices];
  h_sizes_ = new size_t[slices];
  cudaMalloc(&d_ptrs_, sizeof(*d_ptrs_) * slices);
}

BatchDataCompressed::~BatchDataCompressed() {
  delete[] h_ptrs_;
  delete[] h_sizes_;
  cudaFree(d_ptrs_);
  if (compressing_) {
    cudaError_t error{cudaSuccess};
    for (uint64_t i_chunk = 0; error == cudaSuccess && i_chunk < slices_;
         ++i_chunk) {
      error = cudaFree(h_ptrs_[i_chunk]);
    }
  }
}
