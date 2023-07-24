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

  cudaStream_t stream_;
  nvcompStatus_t *statuses_;
  char *device_temporal_memory_;
  BatchDataCompressed *compressed_data_;
  BatchDataUncompressed *uncompressed_data_;

  Opts_t options_;
  size_t chunk_size_;
  size_t batch_size_;
  size_t max_chunk_size_;
  size_t temporal_memory_size_;

  void RemoveTemporalMemory();

  cudaStream_t stream_H2D_;
  cudaStream_t stream_D2H_;
  size_t *host_last_batch_size_;
  size_t *device_last_batch_size_;

  size_t auxiliar_uncompressed_memory_size_;
  size_t auxiliar_compressed_memory_size_;
  size_t number_of_auxiliar_memories_;
  char **list_device_uncompressed_data_;
  char **list_device_compressed_data_;
  size_t **list_device_displacements_;
  BatchDataUncompressed **list_uncompressed_data_;
  BatchDataCompressed **list_compressed_data_;
  cudaEvent_t *list_events_H2D_;
  cudaEvent_t *list_events_D2H_;
  cudaEvent_t *list_events_kernel_;

  cudaError_t InitializeMemories(const size_t &chunk_size,
                                 const size_t &max_chunk_size);

  cudaError_t CompressionH2D(const size_t &batch, const uint16_t &id);

  cudaError_t CompressionD2H(const size_t &batch, const uint16_t &id);

  cudaError_t CompressionH2H(const size_t &batch, const uint16_t &id);

  cudaError_t CompressionMemcpyH2D(size_t *batch, size_t *current_batch_size,
                                   char **host_uncompressed_data,
                                   const uint32_t &uncompressed_data_size,
                                   const uint16_t &id);

  cudaError_t DecompressionMemcpyD2H(char *host_decompressed_data,
                                     uint64_t *decompressed_data_size,
                                     const bool &last_copy,
                                     const uint32_t &batch, const uint16_t &id);

  cudaError_t DecompressionH2D(const size_t &batch, const uint16_t &id);

  cudaError_t DecompressionD2H(const size_t &batch, const uint16_t &id);

  cudaError_t DecompressionH2H(const size_t &batch, const uint16_t &id);

  cudaError_t CompressionMemcpyD2H(char *host_compressed_data,
                                   size_t *host_compressed_displacements,
                                   const size_t &batch, const uint16_t &id,
                                   const bool &no_first_time);

  cudaError_t DecompressionMemcpyH2D(char *host_compressed_data,
                                     size_t *host_compressed_displacements,
                                     const uint32_t &batch, const uint16_t &id,
                                     const bool &no_first_time);

  void GetHeader(size_t *current_batch_size,
                 size_t **device_compressed_displacements,
                 char **device_compressed_data);

  void SetHeader(size_t *current_batch_size,
                 const uint64_t &uncompressed_data_size,
                 size_t **device_compressed_displacements,
                 char **device_compressed_data, uint64_t *compressed_data_size);

  void SetHeaderHost(size_t *current_batch_size,
                     const uint64_t &uncompressed_data_size,
                     size_t **host_compressed_displacements,
                     char **host_compressed_data,
                     uint64_t *compressed_data_size);

  void GetHeaderHost(size_t *current_batch_size,
                     size_t **host_compressed_displacements,
                     char **host_compressed_data);

 public:
  bool CreateInternalStructures(const size_t &batch_size = 0,
                                const uint8_t &streams = 0);

  void DestroyInternalStructures();

  bool InitializeCompression(const size_t &chunk_size,
                             const Opts_t &configuration,
                             const cudaStream_t &stream = nullptr);

  bool InitializeDecompression(const size_t &chunk_size,
                               const cudaStream_t &stream);

  void GetCompressedDataSize(const uint64_t &uncompressed_data_size,
                             uint64_t *compressed_data_size);

  void GetDecompressedDataSize(const char *const device_compressed_data,
                               uint64_t *decompressed_data_size);

  bool CompressDeviceToDevice(const char *const device_uncompressed_data,
                              const uint64_t &uncompressed_data_size,
                              char *device_compressed_data,
                              uint64_t *compressed_data_size);

  bool CompressDeviceToHost(const char *const device_uncompressed_data,
                            const uint64_t &uncompressed_data_size,
                            char *host_compressed_data,
                            uint64_t *compressed_data_size);

  bool CompressHostToDevice(const char *const host_uncompressed_data,
                            const uint64_t &uncompressed_data_size,
                            char *device_compressed_data,
                            uint64_t *compressed_data_size);

  bool CompressHostToHost(const char *const host_uncompressed_data,
                          const uint64_t &uncompressed_data_size,
                          char *host_compressed_data,
                          uint64_t *compressed_data_size);

  bool DecompressDeviceToDevice(const char *const device_compressed_data,
                                const uint64_t &compressed_data_size,
                                char *device_decompressed_data,
                                uint64_t *decompressed_data_size);

  bool DecompressDeviceToHost(const char *const device_compressed_data,
                              const uint64_t &compressed_data_size,
                              char *host_decompressed_data,
                              uint64_t *decompressed_data_size);

  bool DecompressHostToDevice(const char *const host_compressed_data,
                              const uint64_t &compressed_data_size,
                              char *device_decompressed_data,
                              uint64_t *decompressed_data_size);

  bool DecompressHostToHost(const char *const host_compressed_data,
                            const uint64_t &compressed_data_size,
                            char *host_decompressed_data,
                            uint64_t *decompressed_data_size);

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
          cudaStream_t stream));

  ~NvcompTemplate();
};

#include <nvcomp_template_V2.inl>  // NOLINT
