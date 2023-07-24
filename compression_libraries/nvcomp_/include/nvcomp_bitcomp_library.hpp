/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#pragma once

#include <nvcomp/bitcomp.h>

#include <iostream>
#include <string>
#include <vector>

// SMASH LIBRARIES
#include <gpu_compression_library.hpp>
#include <gpu_options.hpp>
#include <nvcomp_template.hpp>

class NvcompBitcompLibrary : public GpuCompressionLibrary {
 private:
  uint8_t number_of_flags_;
  std::string* flags_;

  NvcompTemplate<nvcompBatchedBitcompFormatOpts>* nvcomp_;

  nvcompBatchedBitcompFormatOpts configuration_;

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

  bool GetFlagsInformation(
      std::vector<std::string>* flags_information = nullptr,
      uint8_t* minimum_flags = nullptr, uint8_t* maximum_flags = nullptr);

  bool GetStreamNumberInformation(
      std::vector<std::string>* stream_number_information = nullptr,
      uint8_t* minimum_streams = nullptr, uint8_t* maximum_streams = nullptr);

  std::string GetFlagsName(const uint8_t& flags);

  NvcompBitcompLibrary();
  ~NvcompBitcompLibrary();
};
