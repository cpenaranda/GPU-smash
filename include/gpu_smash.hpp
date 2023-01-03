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
#include <string>
#include <vector>

// SMASH LIBRARIES
#include <gpu_compression_library.hpp>
#include <gpu_options.hpp>

class GpuSmash {
 private:
  GpuCompressionLibrary *lib;

 public:
  bool SetOptionsCompressor(GpuOptions *options, cudaStream_t stream);

  bool SetOptionsDecompressor(GpuOptions *options, cudaStream_t stream);

  void GetCompressedDataSizeFromDeviceMemory(char *device_uncompressed_data,
                                             uint64_t uncompressed_size,
                                             uint64_t *compressed_size);

  void GetCompressedDataSizeFromHostMemory(char *host_uncompressed_data,
                                           uint64_t uncompressed_size,
                                           uint64_t *compressed_size);

  bool CompressDeviceMemory(char *device_uncompressed_data,
                            uint64_t uncompressed_size,
                            char *device_compressed_data,
                            uint64_t *compressed_size);

  bool CompressHostMemory(char *host_uncompressed_data,
                          uint64_t uncompressed_size,
                          char *host_compressed_data,
                          uint64_t *compressed_size);

  void GetDecompressedDataSizeFromDeviceMemory(char *device_compressed_data,
                                               uint64_t compressed_size,
                                               uint64_t *decompressed_size);

  void GetDecompressedDataSizeFromHostMemory(char *host_compressed_data,
                                             uint64_t compressed_size,
                                             uint64_t *decompressed_size);

  bool DecompressDeviceMemory(char *device_compressed_data,
                              uint64_t compressed_size,
                              char *device_decompressed_data,
                              uint64_t *decompressed_size);

  bool DecompressHostMemory(char *host_compressed_data,
                            uint64_t compressed_size,
                            char *host_decompressed_data,
                            uint64_t *decompressed_size);

  void GetTitle();

  bool CompareDataDeviceMemory(char *device_uncompressed_data,
                               const uint64_t &uncompressed_size,
                               char *device_decompressed_data,
                               const uint64_t &decompressed_size);

  bool CompareDataHostMemory(char *host_uncompressed_data,
                             const uint64_t &uncompressed_size,
                             char *host_decompressed_data,
                             const uint64_t &decompressed_size);

  bool GetCompressionLevelInformation(
      std::vector<std::string> *compression_level_information = nullptr,
      uint8_t *minimum_level = nullptr, uint8_t *maximum_level = nullptr);

  bool GetWindowSizeInformation(
      std::vector<std::string> *window_size_information = nullptr,
      uint32_t *minimum_size = nullptr, uint32_t *maximum_size = nullptr);

  bool GetModeInformation(std::vector<std::string> *mode_information = nullptr,
                          uint8_t *minimum_mode = nullptr,
                          uint8_t *maximum_mode = nullptr,
                          const uint8_t &compression_level = 0);

  bool GetWorkFactorInformation(
      std::vector<std::string> *work_factor_information = nullptr,
      uint8_t *minimum_factor = nullptr, uint8_t *maximum_factor = nullptr);

  bool GetFlagsInformation(
      std::vector<std::string> *flags_information = nullptr,
      uint8_t *minimum_flags = nullptr, uint8_t *maximum_flags = nullptr);

  bool GetChunkSizeInformation(
      std::vector<std::string> *chunk_size_information = nullptr,
      uint8_t *minimum_chunk_size = nullptr,
      uint8_t *maximum_chunk_size = nullptr);

  bool GetBackReferenceBitsInformation(
      std::vector<std::string> *back_reference_bits_information = nullptr,
      uint8_t *minimum_bits = nullptr, uint8_t *maximum_bits = nullptr);

  std::string GetModeName(const uint8_t &mode);

  std::string GetFlagsName(const uint8_t &flags);

  GpuOptions GetOptions();

  explicit GpuSmash(const std::string &compression_library_name);

  ~GpuSmash();
};
