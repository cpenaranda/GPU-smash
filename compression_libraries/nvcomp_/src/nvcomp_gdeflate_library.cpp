/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#include <nvcomp/gdeflate.h>

// SMASH LIBRARIES
#include <gpu_options.hpp>
#include <nvcomp_gdeflate_library.hpp>

bool NvcompGdeflateLibrary::CheckOptions(GpuOptions *options,
                                         const bool &compressor) {
  bool result =
      GpuCompressionLibrary::CheckChunkSize("nvcomp-gdeflate", options, 12, 16);
  if (result) {
    if ((result = GpuCompressionLibrary::CheckCompressionLevel(
             "nvcomp-gdeflate", options, 0, 2))) {
      chunk_size_ = 1 << options->GetChunkSize();
      configuration_ = {options->GetCompressionLevel()};
      if (compressor) {
        result = nvcomp_->InitializeCompression(chunk_size_, configuration_,
                                                stream_);
      } else {
        result = nvcomp_->InitializeDecompression(chunk_size_, stream_);
      }
    }
  }
  return result;
}

void NvcompGdeflateLibrary::GetCompressedDataSize(uint64_t uncompressed_size,
                                                  uint64_t *compressed_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetCompressedDataSize(uncompressed_size, compressed_size);
  } else {
    *compressed_size = 0;
  }
}

void NvcompGdeflateLibrary::GetDecompressedDataSizeFromDeviceMemory(
    char *device_compressed_data, uint64_t compressed_size,
    uint64_t *decompressed_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetDecompressedDataSize(device_compressed_data, decompressed_size);
  } else {
    *decompressed_size = 0;
  }
}

bool NvcompGdeflateLibrary::CompressDeviceMemory(char *device_uncompressed_data,
                                                 uint64_t uncompressed_size,
                                                 char *device_compressed_data,
                                                 uint64_t *compressed_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->Compress(device_uncompressed_data, uncompressed_size,
                               device_compressed_data, compressed_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-gdeflate error when compress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompGdeflateLibrary::DecompressDeviceMemory(
    char *device_compressed_data, uint64_t compressed_size,
    char *device_decompressed_data, uint64_t *decompressed_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->Decompress(device_compressed_data, compressed_size,
                                 device_decompressed_data, decompressed_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-gdeflate error when decompress data"
                << std::endl;
    }
  }
  return result;
}

void NvcompGdeflateLibrary::GetTitle() {
  GpuCompressionLibrary::GetTitle(
      "nvcomp-gdeflate",
      "A new compression format that closely matches the DEFLATE format and "
      "allows more efficient GPU decompression.");
}

bool NvcompGdeflateLibrary::GetCompressionLevelInformation(
    std::vector<std::string> *compression_level_information,
    uint8_t *minimum_level, uint8_t *maximum_level) {
  if (minimum_level) *minimum_level = 0;
  if (maximum_level) *maximum_level = 2;
  if (compression_level_information) {
    compression_level_information->clear();
    compression_level_information->push_back("Available values [0-2]");
    compression_level_information->push_back("[compression]");
  }
  return true;
}

bool NvcompGdeflateLibrary::GetChunkSizeInformation(
    std::vector<std::string> *chunk_size_information,
    uint8_t *minimum_chunk_size, uint8_t *maximum_chunk_size) {
  if (minimum_chunk_size) *minimum_chunk_size = 12;
  if (maximum_chunk_size) *maximum_chunk_size = 16;
  if (chunk_size_information) {
    chunk_size_information->clear();
    chunk_size_information->push_back("Available values [12-16]");
    chunk_size_information->push_back("[compression/decompression]");
  }
  return true;
}

NvcompGdeflateLibrary::NvcompGdeflateLibrary(const uint64_t &batch_size) {
  nvcomp_ =
      new NvcompTemplate(nvcompBatchedGdeflateCompressGetTempSize,
                         nvcompBatchedGdeflateCompressGetMaxOutputChunkSize,
                         nvcompBatchedGdeflateDecompressGetTempSize,
                         nvcompBatchedGdeflateGetDecompressSizeAsync,
                         nvcompBatchedGdeflateCompressAsync,
                         nvcompBatchedGdeflateDecompressAsync, batch_size);
}

NvcompGdeflateLibrary::~NvcompGdeflateLibrary() { delete nvcomp_; }
