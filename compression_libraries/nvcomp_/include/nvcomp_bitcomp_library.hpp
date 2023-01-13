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

  bool GetCompressionLevelInformation(
      std::vector<std::string>* compression_level_information = nullptr,
      uint8_t* minimum_level = nullptr, uint8_t* maximum_level = nullptr);

  bool GetFlagsInformation(
      std::vector<std::string>* flags_information = nullptr,
      uint8_t* minimum_flags = nullptr, uint8_t* maximum_flags = nullptr);

  std::string GetFlagsName(const uint8_t& flags);

  NvcompBitcompLibrary(const uint64_t &batch_size = 1000);
  ~NvcompBitcompLibrary();
};
