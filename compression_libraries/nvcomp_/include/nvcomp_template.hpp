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
#include <gpu_compression_library.hpp>
#include <gpu_options.hpp>
#include <nvcomp_batch_compressed.hpp>
#include <nvcomp_batch_uncompressed.hpp>

template <typename Opts_t>
class NvcompTemplate {
 private:
  nvcompStatus_t (*get_temporal_size_to_compress_)(size_t batch_size,
                                                   size_t max_chunk_size,
                                                   Opts_t format_ops,
                                                   size_t *temp_bytes);

  nvcompStatus_t (*get_max_compressed_chunk_size_)(size_t max_chunk_size,
                                                   Opts_t format_opts,
                                                   size_t *max_compressed_size);

  nvcompStatus_t (*get_temporal_size_to_decompress_)(
      size_t num_chunks, size_t max_uncompressed_chunk_size,
      size_t *temp_bytes);
  nvcompStatus_t (*get_decompressed_size_asynchronously_)(
      const void *const *device_compressed_ptrs,
      const size_t *device_compressed_bytes, size_t *device_uncompressed_bytes,
      size_t batch_size, cudaStream_t stream);

  nvcompStatus_t (*compress_asynchronously_)(
      const void *const *device_uncompressed_ptr,
      const size_t *device_uncompressed_bytes,
      size_t max_uncompressed_chunk_bytes, size_t batch_size,
      void *device_temp_ptr, size_t temp_bytes,
      void *const *device_compressed_ptr, size_t *device_compressed_bytes,
      Opts_t format_ops, cudaStream_t stream);

  nvcompStatus_t (*decompress_asynchronously_)(
      const void *const *device_compresed_ptrs,
      const size_t *device_compressed_bytes,
      const size_t *device_uncompressed_bytes,
      size_t *device_actual_uncompressed_bytes, size_t batch_size,
      void *const device_temp_ptr, const size_t temp_bytes,
      void *const *device_uncompressed_ptr, nvcompStatus_t *device_statuses,
      cudaStream_t stream);

  Opts_t options_;

  cudaStream_t stream_;

  size_t chunk_size_;
  size_t batch_size_;
  size_t max_chunk_size_;

  BatchDataUncompressed *uncompressed_data_;
  BatchDataCompressed *compressed_data_;

  char *device_temporal_memory_;
  size_t temporal_memory_size_;
  nvcompStatus_t *statuses;

  void RemoveTemporalMemory();

 public:
  bool InitializeCompression(const size_t &chunk_size,
                             const Opts_t &configuration,
                             const cudaStream_t &stream);

  bool InitializeDecompression(const size_t &chunk_size,
                               const cudaStream_t &stream);

  void GetCompressedDataSize(uint64_t uncompressed_size,
                             uint64_t *compressed_size);

  void GetDecompressedDataSize(char *device_compressed_data,
                               uint64_t *decompressed_size);

  void GetBatchDataInformationFromCompressedData(
      size_t *current_batch_size, size_t **device_compressed_displacements,
      char *device_compressed_data);

  void GetBatchDataInformationFromUncompressedData(
      size_t *current_batch_size, uint64_t uncompressed_size,
      size_t **device_compressed_displacements, char *device_compressed_data,
      uint64_t *compresssed_size);

  bool Compress(char *device_uncompressed_data, uint64_t uncompressed_size,
                char *device_compressed_data, uint64_t *compressed_size);

  bool Decompress(char *device_compressed_data, uint64_t compressed_size,
                  char *device_decompressed_data, uint64_t *decompressed_size);

  NvcompTemplate(
      nvcompStatus_t (*get_temporal_size_to_compress)(size_t batch_size,
                                                      size_t max_chunk_size,
                                                      Opts_t format_ops,
                                                      size_t *temp_bytes),
      nvcompStatus_t (*get_max_compressed_chunk_size)(
          size_t max_chunk_size, Opts_t format_opts,
          size_t *max_compressed_size),
      nvcompStatus_t (*get_temporal_size_to_decompress)(
          size_t num_chunks, size_t max_uncompressed_chunk_size,
          size_t *temp_bytes),
      nvcompStatus_t (*get_decompressed_size_asynchronously)(
          const void *const *device_compressed_ptrs,
          const size_t *device_compressed_bytes,
          size_t *device_uncompressed_bytes, size_t batch_size,
          cudaStream_t stream),
      nvcompStatus_t (*compress_asynchronously)(
          const void *const *device_uncompressed_ptr,
          const size_t *device_uncompressed_bytes,
          size_t max_uncompressed_chunk_bytes, size_t batch_size,
          void *device_temp_ptr, size_t temp_bytes,
          void *const *device_compressed_ptr, size_t *device_compressed_bytes,
          Opts_t format_ops, cudaStream_t stream),
      nvcompStatus_t (*decompress_asynchronously)(
          const void *const *device_compresed_ptrs,
          const size_t *device_compressed_bytes,
          const size_t *device_uncompressed_bytes,
          size_t *device_actual_uncompressed_bytes, size_t batch_size,
          void *const device_temp_ptr, const size_t temp_bytes,
          void *const *device_uncompressed_ptr, nvcompStatus_t *device_statuses,
          cudaStream_t stream),
      const size_t &batch_size = 1000);

  ~NvcompTemplate();
};

#include <nvcomp_template.inl>  // NOLINT
