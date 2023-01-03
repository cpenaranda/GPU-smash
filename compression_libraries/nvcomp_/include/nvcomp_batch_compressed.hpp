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

class BatchDataCompressed {
 public:
  // Compressing
  cudaError_t InitilizeCompression(char *data, size_t *sizes,
                                   const size_t &max_chunk_size,
                                   const cudaStream_t &stream);

  cudaError_t DumpData(const size_t &chunks, const cudaStream_t &stream);

  // Decompressing
  cudaError_t InitilizeDecompression(char *data, const size_t &data_size,
                                     size_t *sizes);

  cudaError_t GetNext(const size_t &chunks, const cudaStream_t &stream);

  cudaError_t IncrementSizes(const size_t &chunks);

  void *const *d_ptrs();

  size_t *d_sizes();

  size_t size();

  BatchDataCompressed(const size_t &slices);

  ~BatchDataCompressed();

 private:
  char *data_;
  size_t size_;
  size_t slices_;
  char **d_ptrs_;
  char **h_ptrs_;
  size_t *d_sizes_;
  size_t *h_sizes_;
  bool compressing_;
};

#include <nvcomp_batch_compressed.inl>  // NOLINT
