/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#include <nvcomp/ans.h>

// SMASH LIBRARIES
#include <gpu_options.hpp>
#include <nvcomp_ans_library.hpp>

bool NvcompAnsLibrary::CheckOptions(GpuOptions *options,
                                    const bool &compressor) {
  bool result{true};
  configuration_ = {nvcomp_rANS};
  return result;
}

void NvcompAnsLibrary::GetCompressedDataSize(uint64_t uncompressed_size,
                                             uint64_t *compressed_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->InitializeCompression(uncompressed_size, configuration_, stream_);
    nvcomp_->GetCompressedDataSize(uncompressed_size, compressed_size);
  } else {
    *compressed_size = 0;
  }
}

void NvcompAnsLibrary::GetDecompressedDataSizeFromDeviceMemory(
    char *device_compressed_data, uint64_t compressed_size,
    uint64_t *decompressed_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetDecompressedDataSize(device_compressed_data, decompressed_size);
  } else {
    *decompressed_size = 0;
  }
}

bool NvcompAnsLibrary::CompressDeviceMemory(char *device_uncompressed_data,
                                            uint64_t uncompressed_size,
                                            char *device_compressed_data,
                                            uint64_t *compressed_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->InitializeCompression(uncompressed_size, configuration_,
                                            stream_);
    if (result) {
      result = nvcomp_->Compress(device_uncompressed_data, uncompressed_size,
                                 device_compressed_data, compressed_size);
      if (!result) {
        std::cout << "ERROR: nvcomp-ans error when compress data" << std::endl;
      }
    }
  }
  return result;
}

bool NvcompAnsLibrary::DecompressDeviceMemory(char *device_compressed_data,
                                              uint64_t compressed_size,
                                              char *device_decompressed_data,
                                              uint64_t *decompressed_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->InitializeDecompression(*decompressed_size, stream_);
    if (result) {
      result = nvcomp_->Decompress(device_compressed_data, compressed_size,
                                   device_decompressed_data, decompressed_size);
      if (!result) {
        std::cout << "ERROR: nvcomp-ans error when decompress data"
                  << std::endl;
      }
    }
  }
  return result;
}

void NvcompAnsLibrary::GetTitle() {
  GpuCompressionLibrary::GetTitle(
      "nvcomp-ans",
      "Proprietary entropy encoder based on asymmetric numeral systems.");
}

NvcompAnsLibrary::NvcompAnsLibrary(const uint64_t &batch_size) {
  nvcomp_ = new NvcompTemplate(nvcompBatchedANSCompressGetTempSize,
                               nvcompBatchedANSCompressGetMaxOutputChunkSize,
                               nvcompBatchedANSDecompressGetTempSize,
                               nvcompBatchedANSGetDecompressSizeAsync,
                               nvcompBatchedANSCompressAsync,
                               nvcompBatchedANSDecompressAsync, 1);
}

NvcompAnsLibrary::~NvcompAnsLibrary() { delete nvcomp_; }
