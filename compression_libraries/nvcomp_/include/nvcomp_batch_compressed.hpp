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

class BatchDataCompressed {
 public:
  // Compressing
  cudaError_t InitializeCompression(char *data, size_t *displacements,
                                    const cudaStream_t &stream);

  cudaError_t ConfigureCompression(const size_t &max_chunk_size);

  cudaError_t DumpData(const size_t &chunks, const cudaStream_t &stream);

  cudaError_t DumpDataPipeline(size_t *device_last_batch_size,
                               const size_t &chunks,
                               const cudaStream_t &stream);

  void DumpDataPipeline2(const size_t &chunks, const cudaStream_t &stream);

  cudaError_t IncrementPipeline(size_t *device_last_batch_size,
                                const size_t &chunks,
                                const cudaStream_t &stream);

  cudaError_t IncrementPipeline(const size_t &last_batch_size,
                                const size_t &chunks,
                                const cudaStream_t &stream);
  // Decompressing
  cudaError_t InitializeDecompression(const char *const data,
                                      size_t *displacements,
                                      const cudaStream_t &stream);

  cudaError_t GetNext(const size_t &chunks, const cudaStream_t &stream);

  cudaError_t GetNextPipeline(const size_t &chunks, const cudaStream_t &stream,
                              const size_t &previous_displacement);

  void *const *d_ptrs();

  size_t *d_sizes();

  void GetSize(size_t *size);

  BatchDataCompressed(const size_t &slices);

  ~BatchDataCompressed();

 private:
  char *data_;
  size_t *d_size_;
  size_t slices_;
  char **d_ptrs_;
  size_t *d_sizes_;
  size_t *d_displacements_;
  size_t last_chunk_;

  // Compressing
  char **h_ptrs_compression_;
  size_t max_chunk_size_;
};

#include <nvcomp_batch_compressed.inl>  // NOLINT
