/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#pragma once

#include <assert.h>
#include <dietgpu/ans/GpuANSCodec.h>
#include <dietgpu/utils/StackDeviceMemory.h>

#include <iostream>
#include <string>
#include <vector>

// SMASH LIBRARIES
#include <gpu_compression_library.hpp>
#include <gpu_options.hpp>

class DietgpuAnsLibrary : public GpuCompressionLibrary {
 private:
  uint32_t uncompressed_chunk_size_;
  uint32_t compressed_chunk_size_;
  dietgpu::StackDeviceMemory* temp_memory_;
  dietgpu::ANSCodecConfig configuration_ANS_;

  bool reserved_compressed_temporal_memory_;
  uint64_t batch_size_;
  void** host_uncompressed_ptrs_;
  uint32_t* device_uncompressed_sizes_;
  uint32_t* host_uncompressed_sizes_;
  void** host_compressed_ptrs_;
  uint32_t* host_compressed_sizes_;

 public:
  bool CheckOptions(GpuOptions* options, const bool& compressor);

  void GetCompressedDataSize(uint64_t uncompressed_size,
                             uint64_t* compressed_size);

  void GetDecompressedDataSizeFromDeviceMemory(char* device_compressed_data,
                                               uint64_t compressed_size,
                                               uint64_t* decompressed_size);

  bool CompressDeviceMemory(char* device_uncompressed_data,
                            uint64_t uncompressed_size,
                            char* device_compressed_data,
                            uint64_t* compressed_size);

  bool DecompressDeviceMemory(char* device_compressed_data,
                              uint64_t compressed_size,
                              char* device_decompressed_data,
                              uint64_t* decompressed_size);

  void GetTitle();

  bool GetChunkSizeInformation(
      std::vector<std::string>* chunk_size_information = nullptr,
      uint8_t* minimum_chunk_size = nullptr,
      uint8_t* maximum_chunk_size = nullptr);

  bool GetCompressionLevelInformation(
      std::vector<std::string>* compression_level_information = nullptr,
      uint8_t* minimum_level = nullptr, uint8_t* maximum_level = nullptr);

  DietgpuAnsLibrary(const uint64_t& batch_size = 1000);
  ~DietgpuAnsLibrary();
};
