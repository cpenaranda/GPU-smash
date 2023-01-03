/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#pragma once

#include <nvcomp/zstd.h>

#include <iostream>
#include <string>
#include <vector>

// SMASH LIBRARIES
#include <gpu_compression_library.hpp>
#include <gpu_options.hpp>
#include <nvcomp_template.hpp>

class NvcompZstdLibrary : public GpuCompressionLibrary {
 private:
  NvcompTemplate<nvcompBatchedZstdOpts_t>* nvcomp_;

  size_t chunk_size_;
  nvcompBatchedZstdOpts_t configuration_;

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

  NvcompZstdLibrary();
  ~NvcompZstdLibrary();
};
