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
#include <nvcomp_batch_uncompressed.hpp>

BatchDataUncompressed::BatchDataUncompressed(const size_t &slices)
    : slices_(slices) {
  h_ptrs_ = new char *[slices];
  h_sizes_ = new size_t[slices];
  cudaMalloc(&d_ptrs_, sizeof(*d_ptrs_) * slices);
  cudaMalloc(&d_sizes_, sizeof(*d_sizes_) * slices_);
}

BatchDataUncompressed::~BatchDataUncompressed() {
  delete[] h_ptrs_;
  delete[] h_sizes_;
  cudaFree(d_ptrs_);
  cudaFree(d_sizes_);
}
