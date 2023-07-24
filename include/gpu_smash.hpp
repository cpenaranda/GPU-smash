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

// GPU-SMASH LIBRARIES
#include <gpu_compression_library.hpp>
#include <gpu_options.hpp>

enum class CompressionType {
  DeviceToDevice,
  DeviceToHost,
  HostToDevice,
  HostToHost,
};

class GpuSmash {
 private:
  GpuCompressionLibrary *library_;

 public:
  bool SetOptionsCompressor(GpuOptions *options, const cudaStream_t &stream);

  bool SetOptionsDecompressor(GpuOptions *options, const cudaStream_t &stream);

  void GetCompressedDataSizeFromDeviceMemory(
      const char *const device_uncompressed_data,
      const uint64_t &uncompressed_data_size, uint64_t *compressed_data_size);

  void GetCompressedDataSizeFromHostMemory(
      const char *const host_uncompressed_data,
      const uint64_t &uncompressed_data_size, uint64_t *compressed_data_size);

  bool Compress(const char *const uncompressed_data,
                const uint64_t &uncompressed_data_size, char *compressed_data,
                uint64_t *compressed_data_size, const CompressionType &type);

  void GetDecompressedDataSizeFromDeviceMemory(
      const char *const device_compressed_data,
      const uint64_t &compressed_data_size, uint64_t *decompressed_data_size);

  void GetDecompressedDataSizeFromHostMemory(
      const char *const host_compressed_data,
      const uint64_t &compressed_data_size, uint64_t *decompressed_data_size);

  bool Decompress(const char *const compressed_data,
                  const uint64_t &compressed_data_size, char *decompressed_data,
                  uint64_t *decompressed_data_size,
                  const CompressionType &type);

  void GetTitle();

  bool CompareDataDeviceMemory(const char *const device_uncompressed_data,
                               const uint64_t &uncompressed_data_size,
                               const char *const device_decompressed_data,
                               const uint64_t &decompressed_data_size);

  bool CompareDataHostMemory(const char *const host_uncompressed_data,
                             const uint64_t &uncompressed_data_size,
                             const char *const host_decompressed_data,
                             const uint64_t &decompressed_data_size);

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

  bool GetChunkNumberInformation(
      std::vector<std::string> *chunk_number_information = nullptr,
      uint8_t *minimum_chunks = nullptr, uint8_t *maximum_chunks = nullptr);

  bool GetStreamNumberInformation(
      std::vector<std::string> *stream_number_information = nullptr,
      uint8_t *minimum_streams = nullptr, uint8_t *maximum_streams = nullptr);

  bool GetBackReferenceInformation(
      std::vector<std::string> *back_reference_information = nullptr,
      uint8_t *minimum_back_reference = nullptr,
      uint8_t *maximum_back_reference = nullptr);

  std::string GetModeName(const uint8_t &mode);

  std::string GetFlagsName(const uint8_t &flags);

  GpuOptions GetOptions();

  explicit GpuSmash(const std::string &compression_library_name);

  ~GpuSmash();
};
