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

class NvcompUtil {
 private:
  static constexpr const size_t blocks_[12] = {2048, 1024, 512, 256, 128, 64,
                                               32,   16,   8,   4,   2,   1};
  static constexpr const int threads_[12] = {64,  128, 128, 256, 512, 512,
                                             512, 512, 256, 256, 128, 1024};

 public:
  static void DumpData(char *data, char **ptrs, size_t *size, size_t *sizes,
                       size_t *displacements, const size_t &chunks,
                       const cudaStream_t &stream);

  static void DumpDataPipeline(char *data, char **ptrs, size_t *inc,
                               size_t *sizes, size_t *displacements,
                               const size_t &chunks,
                               const cudaStream_t &stream);

  static void DumpDataPipeline2(char *data, char **ptrs, size_t *sizes,
                                size_t *displacements, const size_t &chunks,
                                const cudaStream_t &stream);

  static void IncrementPipeline(size_t *inc, size_t *displacements,
                                const size_t &chunks,
                                const cudaStream_t &stream);

  static void IncrementPipeline(const size_t &inc, size_t *displacements,
                                const size_t &chunks,
                                const cudaStream_t &stream);

  static void GetNext(char *data, char **ptrs, size_t *sizes,
                      const size_t &chunk_size, const size_t &chunks,
                      const cudaStream_t &stream);

  static void GetNext(char *data, char **ptrs, size_t *size, size_t *sizes,
                      size_t *displacements, const size_t &chunks,
                      const cudaStream_t &stream);

  static void GetNextPipeline(char *data, char **ptrs, const size_t &dec,
                              size_t *sizes, size_t *displacements,
                              const size_t &chunks, const cudaStream_t &stream);

  static void DumpData(char *dst, char *src, size_t *size,
                       size_t *displacements, size_t *previous_displacement,
                       const size_t &stream_chunk, const cudaStream_t &stream);
};
