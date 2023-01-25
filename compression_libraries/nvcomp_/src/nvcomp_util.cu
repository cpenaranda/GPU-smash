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
#include <nvcomp_util.cuh>

__global__ void DumpDataKernel(char *data, char **ptrs, size_t *size,
                               size_t *sizes, size_t *displacement) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  char *pointer_to_copy = ptrs[bx];
  char *begin = data;

  data += *size;
  for (int i = 0; i < bx; ++i) {
    data += sizes[i];
  }
  if (tx == 0) { displacement[bx] = (data - begin) + sizes[bx]; }
  while (tx < sizes[bx]) {
    data[tx] = pointer_to_copy[tx];
    tx += blockDim.x;
  }
}

void NvcompUtil::DumpData(char *data, char **ptrs, size_t *size,
                          size_t *sizes, size_t *displacements,
                          const size_t &chunks, const cudaStream_t &stream) {
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  DumpDataKernel<<<chunks, prop.maxThreadsPerBlock, 0, stream>>>(data, ptrs, size, sizes, displacements);
}

__global__ void GetNextUncompressedDataKernel(char *data, char **ptrs,
                                              size_t chunk_size,
                                              size_t *sizes) {
  int id = blockIdx.x;
  ptrs[id] = data + (id * chunk_size);
  sizes[id] = chunk_size;
}

void NvcompUtil::GetNext(char *data, char **ptrs, size_t *sizes,
                         const size_t &chunk_size, const size_t &chunks,
                         const cudaStream_t &stream) {
  GetNextUncompressedDataKernel<<<chunks, 1, 0, stream>>>(data, ptrs,
                                                          chunk_size, sizes);
}

__global__ void GetNextCompressedDataKernel(char *data, char **ptrs,
                                            size_t *size, size_t *sizes,
                                            size_t *displacements) {
  int id = blockIdx.x;
  size_t previous_displacement = 0;
  if (id != 0 || *size != 0) {
    previous_displacement = *(displacements + id - 1);
  }
  ptrs[id] = data + previous_displacement;
  sizes[id] = displacements[id] - previous_displacement;
}

void NvcompUtil::GetNext(char *data, char **ptrs, size_t *size, size_t *sizes,
                         size_t *displacements, const size_t &chunks,
                         const cudaStream_t &stream) {
  GetNextCompressedDataKernel<<<chunks, 1, 0, stream>>>(data, ptrs, size, sizes,
                                                        displacements);
}
