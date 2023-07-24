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
  cudaError_t InitializeCompression(const char *const data,
                                    const size_t &data_size);

  cudaError_t GetNextCompression(const size_t &chunks, const size_t &chunk_size,
                                 const cudaStream_t &stream);

  // Decompressing
  cudaError_t InitializeDecompression(char *data, const cudaStream_t &stream);

  cudaError_t GetNextDecompression(const size_t &chunks,
                                   const size_t &chunk_size,
                                   const cudaStream_t &stream);

  void *const *d_ptrs();

  size_t *d_sizes();

  size_t size(const size_t &last_chunk = 0);

  BatchDataUncompressed(const size_t &slices);

  ~BatchDataUncompressed();

 private:
  char *data_;
  size_t slices_;
  char **d_ptrs_;
  size_t *d_sizes_;

  size_t size_;

  // Decompressing
  char *begin_data_;
  size_t last_chunk_;
};

#include <nvcomp_batch_uncompressed.inl>  // NOLINT
