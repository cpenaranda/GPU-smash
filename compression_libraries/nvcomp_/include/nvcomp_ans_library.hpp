/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#pragma once

#include <nvcomp/ans.h>

#include <iostream>

// SMASH LIBRARIES
#include <gpu_compression_library.hpp>
#include <gpu_options.hpp>
#include <nvcomp_template.hpp>

class NvcompAnsLibrary : public GpuCompressionLibrary {
 private:
  NvcompTemplate<nvcompBatchedANSOpts_t>* nvcomp_;

  nvcompBatchedANSOpts_t configuration_;

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

  NvcompAnsLibrary(const uint64_t& batch_size = 1000);
  ~NvcompAnsLibrary();
};
