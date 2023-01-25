/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#pragma once

#include <cuda_runtime.h>

#include <iostream>

// SMASH LIBRARIES
#include <nvcomp_util.cuh>

class BatchDataUncompressed {
 public:
  // Compressing
  cudaError_t InitilizeCompression(char *data, const size_t &data_size);

  cudaError_t GetNextCompression(const size_t &chunks, const size_t &chunk_size,
                                 const cudaStream_t &stream);

  // Decompressing
  cudaError_t InitilizeDecompression(char *data, const cudaStream_t &stream);

  cudaError_t GetNextDecompression(const size_t &chunks,
                                   const size_t &chunk_size,
                                   const cudaStream_t &stream);

  void *const *d_ptrs();

  size_t *d_sizes();

  size_t size();

  BatchDataUncompressed(const size_t &slices);

  ~BatchDataUncompressed();

 private:
  char *data_;
  size_t slices_;
  char **d_ptrs_;
  size_t *d_sizes_;

  // Compressing
  size_t size_;

  // Decompressing
  char *begin_data_;
  size_t chunk_size_;
  size_t last_chunk_;
};

#include <nvcomp_batch_uncompressed.inl>  // NOLINT
