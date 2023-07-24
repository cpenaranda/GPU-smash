/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#include <cuda_runtime.h>

#include <chrono>  // NOLINT
#include <iostream>

// SMASH LIBRARIES
#include <nvcomp_util.cuh>

inline __device__ void copy(char *dst, char *src, const size_t &size, int id,
                            const int &jump) {
  while (id < size) {
    dst[id] = src[id];
    id += jump;
  }
}

__global__ void DumpDataKernel(char *data, char **ptrs, size_t *size,
                               size_t *sizes, size_t *displacement) {
  int bx = blockIdx.x;
  char *begin = data;
  data += *size;
  for (int i = 0; i < bx; ++i) {
    data += sizes[i];
  }
  if (threadIdx.x == 0) {
    displacement[bx] = (data - begin) + sizes[bx];
  }
  copy(data, ptrs[bx], sizes[bx], threadIdx.x, blockDim.x);
}

void NvcompUtil::DumpData(char *data, char **ptrs, size_t *size, size_t *sizes,
                          size_t *displacements, const size_t &chunks,
                          const cudaStream_t &stream) {
  size_t blocks_[12] = {2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1};
  int threads_[12] = {64,  128, 128, 256, 512, 512,
                      512, 512, 256, 256, 128, 1024};
  int i_block;
  for (size_t current_chunk = 0; current_chunk < chunks;
       current_chunk += blocks_[i_block]) {
    i_block = 0;
    for (const auto &block : blocks_) {
      if (block + current_chunk > chunks)
        ++i_block;
      else
        break;
    }
    DumpDataKernel<<<blocks_[i_block], threads_[i_block], 0, stream>>>(
        data, ptrs + current_chunk, size, sizes + current_chunk,
        displacements + current_chunk);
    cudaMemcpyAsync(size, displacements + current_chunk + blocks_[i_block] - 1,
                    sizeof(*size), cudaMemcpyDeviceToDevice, stream);
  }
}

__global__ void DumpDataKernelPipeline(char *data, char **ptrs,
                                       size_t current_chunk, size_t *inc,
                                       size_t *sizes, size_t *displacement) {
  int bx = blockIdx.x;
  char *begin = data;
  data += (current_chunk ? (*(displacement - 1) - *inc) : 0);
  for (int i = 0; i < bx; ++i) {
    data += sizes[i];
  }
  if (threadIdx.x == 0) {
    displacement[bx] = (data - begin) + sizes[bx] + *inc;
  }
  copy(data, ptrs[bx], sizes[bx], threadIdx.x, blockDim.x);
}

void NvcompUtil::DumpDataPipeline(char *data, char **ptrs, size_t *inc,
                                  size_t *sizes, size_t *displacements,
                                  const size_t &chunks,
                                  const cudaStream_t &stream) {
  size_t blocks_[12] = {2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1};
  int threads_[12] = {64,  128, 128, 256, 512, 512,
                      512, 512, 256, 256, 128, 1024};
  int i_block;
  for (size_t current_chunk = 0; current_chunk < chunks;
       current_chunk += blocks_[i_block]) {
    i_block = 0;
    for (const auto &block : blocks_) {
      if (block + current_chunk > chunks)
        ++i_block;
      else
        break;
    }
    DumpDataKernelPipeline<<<blocks_[i_block], threads_[i_block], 0, stream>>>(
        data, ptrs + current_chunk, current_chunk, inc, sizes + current_chunk,
        displacements + current_chunk);
  }
}

__global__ void DumpDataKernelPipeline2(char *data, char **ptrs,
                                        size_t current_chunk, size_t *sizes,
                                        size_t *displacement) {
  int bx = blockIdx.x;
  char *begin = data;
  data += (current_chunk ? *(displacement - 1) : 0);
  for (int i = 0; i < bx; ++i) {
    data += sizes[i];
  }
  if (threadIdx.x == 0) {
    displacement[bx] = (data - begin) + sizes[bx];
  }
  copy(data, ptrs[bx], sizes[bx], threadIdx.x, blockDim.x);
}

void NvcompUtil::DumpDataPipeline2(char *data, char **ptrs, size_t *sizes,
                                   size_t *displacements, const size_t &chunks,
                                   const cudaStream_t &stream) {
  size_t blocks_[12] = {2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1};
  int threads_[12] = {64,  128, 128, 256, 512, 512,
                      512, 512, 256, 256, 128, 1024};
  int i_block;
  for (size_t current_chunk = 0; current_chunk < chunks;
       current_chunk += blocks_[i_block]) {
    i_block = 0;
    for (const auto &block : blocks_) {
      if (block + current_chunk > chunks)
        ++i_block;
      else
        break;
    }
    DumpDataKernelPipeline2<<<blocks_[i_block], threads_[i_block], 0, stream>>>(
        data, ptrs + current_chunk, current_chunk, sizes + current_chunk,
        displacements + current_chunk);
  }
}

__global__ void IncrementKernelPipelineDevice(size_t *inc,
                                              size_t *displacement) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  displacement[id] += *inc;
}

__global__ void IncrementKernelPipelineHost(size_t inc, size_t *displacement) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  printf("PRE(%d) %lu\n", id, displacement[id]);
  displacement[id] += inc;
  printf("AFT(%d) %lu\n", id, displacement[id]);
}

void NvcompUtil::IncrementPipeline(size_t *inc, size_t *displacements,
                                   const size_t &chunks,
                                   const cudaStream_t &stream) {
  int threads = 256;
  for (size_t current_chunk = 0, blocks = 0; current_chunk < chunks;
       current_chunk += (blocks * threads)) {
    blocks = (chunks - current_chunk) / threads;
    if (!blocks) {
      threads = (chunks - current_chunk);
      blocks = 1;
    }
    IncrementKernelPipelineDevice<<<blocks, threads, 0, stream>>>(
        inc, displacements);
  }
}

void NvcompUtil::IncrementPipeline(const size_t &inc, size_t *displacements,
                                   const size_t &chunks,
                                   const cudaStream_t &stream) {
  int threads = 256;
  for (size_t current_chunk = 0, blocks = 0; current_chunk < chunks;
       current_chunk += (blocks * threads)) {
    blocks = (chunks - current_chunk) / threads;
    if (!blocks) {
      threads = (chunks - current_chunk);
      blocks = 1;
    }
    IncrementKernelPipelineHost<<<blocks, threads, 0, stream>>>(inc,
                                                                displacements);
  }
}

__global__ void GetNextUncompressedDataKernel(char *data, char **ptrs,
                                              size_t chunk_size,
                                              size_t *sizes) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  ptrs[id] = data + (id * chunk_size);
  sizes[id] = chunk_size;
}

void NvcompUtil::GetNext(char *data, char **ptrs, size_t *sizes,
                         const size_t &chunk_size, const size_t &chunks,
                         const cudaStream_t &stream) {
  int threads = 256;
  for (size_t current_chunk = 0, blocks = 0; current_chunk < chunks;
       current_chunk += (blocks * threads)) {
    blocks = (chunks - current_chunk) / threads;
    if (!blocks) {
      threads = (chunks - current_chunk);
      blocks = 1;
    }
    GetNextUncompressedDataKernel<<<blocks, threads, 0, stream>>>(
        data + current_chunk * chunk_size, ptrs + current_chunk, chunk_size,
        sizes + current_chunk);
  }
}

__global__ void GetNextCompressedDataKernel(char *data, char **ptrs,
                                            size_t *size, size_t *sizes,
                                            size_t *displacements) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t previous_displacement =
      (id != 0 || *size != 0) ? *(displacements + id - 1) : 0;
  ptrs[id] = data + previous_displacement;
  sizes[id] = displacements[id] - previous_displacement;
}

void NvcompUtil::GetNext(char *data, char **ptrs, size_t *size, size_t *sizes,
                         size_t *displacements, const size_t &chunks,
                         const cudaStream_t &stream) {
  int threads = 256;
  for (size_t current_chunk = 0, blocks = 0; current_chunk < chunks;
       current_chunk += (blocks * threads)) {
    blocks = (chunks - current_chunk) / threads;
    if (!blocks) {
      threads = (chunks - current_chunk);
      blocks = 1;
    }
    GetNextCompressedDataKernel<<<blocks, threads, 0, stream>>>(
        data, ptrs + current_chunk, size, sizes + current_chunk,
        displacements + current_chunk);
    cudaMemcpyAsync(size,
                    displacements + current_chunk + (blocks * threads) - 1,
                    sizeof(*size), cudaMemcpyDeviceToDevice, stream);
  }
}

__global__ void GetNextCompressedDataKernelPipeline(char *data, char **ptrs,
                                                    size_t current_chunk,
                                                    size_t dec, size_t *sizes,
                                                    size_t *displacements) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id != 0 || current_chunk) {
    size_t previous_displacement = *(displacements + id - 1);
    ptrs[id] = data + previous_displacement - dec;
    sizes[id] = displacements[id] - previous_displacement;
  } else {
    ptrs[id] = data;
    sizes[id] = displacements[id] - dec;
  }
}

void NvcompUtil::GetNextPipeline(char *data, char **ptrs, const size_t &dec,
                                 size_t *sizes, size_t *displacements,
                                 const size_t &chunks,
                                 const cudaStream_t &stream) {
  int threads = 256;
  for (size_t current_chunk = 0, blocks = 0; current_chunk < chunks;
       current_chunk += (blocks * threads)) {
    blocks = (chunks - current_chunk) / threads;
    if (!blocks) {
      threads = (chunks - current_chunk);
      blocks = 1;
    }
    GetNextCompressedDataKernelPipeline<<<blocks, threads, 0, stream>>>(
        data, ptrs + current_chunk, current_chunk, dec, sizes + current_chunk,
        displacements + current_chunk);
  }
}

__global__ void DumpDataAndFixDisplacementsKernel(
    char *dst, char *src, size_t *size, size_t *displacements,
    size_t *previous_displacement) {
  if (threadIdx.x == 0) {
    displacements[blockIdx.x] += *previous_displacement;
  }
  copy(dst + *previous_displacement, src, *size,
       blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x);
}

void NvcompUtil::DumpData(char *dst, char *src, size_t *size,
                          size_t *displacements, size_t *previous_displacement,
                          const size_t &stream_chunk,
                          const cudaStream_t &stream) {
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  DumpDataAndFixDisplacementsKernel<<<stream_chunk, prop.maxThreadsPerBlock, 0,
                                      stream>>>(dst, src, size, displacements,
                                                previous_displacement);
}
