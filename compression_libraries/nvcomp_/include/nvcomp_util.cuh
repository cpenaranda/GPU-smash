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
 public:
  static void DumpData(char *data, char **ptrs, size_t *size,
                       size_t *sizes, size_t *displacements,
                       const size_t &chunks, const cudaStream_t &stream);

  static void GetNext(char *data, char **ptrs, size_t *sizes,
                      const size_t &chunk_size, const size_t &chunks,
                      const cudaStream_t &stream);

  static void GetNext(char *data, char **ptrs, size_t *size, size_t *sizes,
                      size_t *displacements, const size_t &chunks,
                      const cudaStream_t &stream);
};
