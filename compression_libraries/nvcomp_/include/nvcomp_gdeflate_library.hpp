/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#pragma once

#include <nvcomp/gdeflate.h>

#include <iostream>
#include <string>
#include <vector>

// SMASH LIBRARIES
#include <gpu_compression_library.hpp>
#include <gpu_options.hpp>
#include <nvcomp_template.hpp>

class NvcompGdeflateLibrary : public GpuCompressionLibrary {
 private:
  NvcompTemplate<nvcompBatchedGdeflateOpts_t>* nvcomp_;

  size_t chunk_size_;
  nvcompBatchedGdeflateOpts_t configuration_;

 public:
  bool CheckOptions(GpuOptions* options, const bool& compressor);

  void GetCompressedDataSize(const uint64_t& uncompressed_data_size,
                             uint64_t* compressed_sdata_ize);

  void GetDecompressedDataSizeFromDeviceMemory(
      const char* const device_compressed_data,
      const uint64_t& compressed_data_size, uint64_t* decompressed_data_size);

  bool CompressDeviceToDevice(const char* const device_uncompressed_data,
                              const uint64_t& uncompressed_data_size,
                              char* device_compressed_data,
                              uint64_t* compressed_data_size);

  bool CompressHostToDevice(const char* const host_uncompressed_data,
                            const uint64_t& uncompressed_data_size,
                            char* device_compressed_data,
                            uint64_t* compressed_data_size);

  bool CompressDeviceToHost(const char* const device_uncompressed_data,
                            const uint64_t& uncompressed_data_size,
                            char* host_compressed_data,
                            uint64_t* compressed_data_size);

  bool CompressHostToHost(const char* const host_uncompressed_data,
                          const uint64_t& uncompressed_data_size,
                          char* host_compressed_data,
                          uint64_t* compressed_data_size);

  bool DecompressDeviceToDevice(const char* const device_compressed_data,
                                const uint64_t& compressed_data_size,
                                char* device_decompressed_data,
                                uint64_t* decompressed_data_size);

  bool DecompressDeviceToHost(const char* const device_compressed_data,
                              const uint64_t& compressed_data_size,
                              char* host_decompressed_data,
                              uint64_t* decompressed_data_size);

  bool DecompressHostToDevice(const char* const host_compressed_data,
                              const uint64_t& compressed_data_size,
                              char* device_decompressed_data,
                              uint64_t* decompressed_data_size);

  bool DecompressHostToHost(const char* const host_compressed_data,
                            const uint64_t& compressed_data_size,
                            char* host_decompressed_data,
                            uint64_t* decompressed_data_size);

  void GetTitle();

  bool GetCompressionLevelInformation(
      std::vector<std::string>* compression_level_information = nullptr,
      uint8_t* minimum_level = nullptr, uint8_t* maximum_level = nullptr);

  bool GetChunkSizeInformation(
      std::vector<std::string>* chunk_size_information = nullptr,
      uint8_t* minimum_chunk_size = nullptr,
      uint8_t* maximum_chunk_size = nullptr);

  bool GetChunkNumberInformation(
      std::vector<std::string>* chunk_number_information = nullptr,
      uint8_t* minimum_chunks = nullptr, uint8_t* maximum_chunks = nullptr);

  bool GetStreamNumberInformation(
      std::vector<std::string>* stream_number_information = nullptr,
      uint8_t* minimum_streams = nullptr, uint8_t* maximum_streams = nullptr);

  NvcompGdeflateLibrary();
  ~NvcompGdeflateLibrary();
};
