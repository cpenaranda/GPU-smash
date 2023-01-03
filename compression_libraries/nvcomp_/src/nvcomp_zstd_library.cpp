/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#include <nvcomp/zstd.h>

// SMASH LIBRARIES
#include <gpu_options.hpp>
#include <nvcomp_zstd_library.hpp>

bool NvcompZstdLibrary::CheckOptions(GpuOptions *options,
                                     const bool &compressor) {
  bool result =
      GpuCompressionLibrary::CheckChunkSize("nvcomp-zstd", options, 12, 24);
  if (result) {
    chunk_size_ = 1 << options->GetChunkSize();
    if (compressor) {
      configuration_ = {0};
      result =
          nvcomp_->InitializeCompression(chunk_size_, configuration_, stream_);
    } else {
      result = nvcomp_->InitializeDecompression(chunk_size_, stream_);
    }
  }
  return result;
}

void NvcompZstdLibrary::GetCompressedDataSize(uint64_t uncompressed_size,
                                              uint64_t *compressed_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetCompressedDataSize(uncompressed_size, compressed_size);
  } else {
    *compressed_size = 0;
  }
}

void NvcompZstdLibrary::GetDecompressedDataSizeFromDeviceMemory(
    char *device_compressed_data, uint64_t compressed_size,
    uint64_t *decompressed_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetDecompressedDataSize(device_compressed_data, decompressed_size);
  } else {
    *decompressed_size = 0;
  }
}

bool NvcompZstdLibrary::CompressDeviceMemory(char *device_uncompressed_data,
                                             uint64_t uncompressed_size,
                                             char *device_compressed_data,
                                             uint64_t *compressed_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->Compress(device_uncompressed_data, uncompressed_size,
                               device_compressed_data, compressed_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-zstd error when compress data" << std::endl;
    }
  }
  return result;
}

bool NvcompZstdLibrary::DecompressDeviceMemory(char *device_compressed_data,
                                               uint64_t compressed_size,
                                               char *device_decompressed_data,
                                               uint64_t *decompressed_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->Decompress(device_compressed_data, compressed_size,
                                 device_decompressed_data, decompressed_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-zstd error when decompress data" << std::endl;
    }
  }
  return result;
}

void NvcompZstdLibrary::GetTitle() {
  GpuCompressionLibrary::GetTitle(
      "nvcomp-zstd",
      "Huffman + LZ77 + ANS, popular compression format developed by Meta.");
}

bool NvcompZstdLibrary::GetChunkSizeInformation(
    std::vector<std::string> *chunk_size_information,
    uint8_t *minimum_chunk_size, uint8_t *maximum_chunk_size) {
  if (minimum_chunk_size) *minimum_chunk_size = 12;
  if (maximum_chunk_size) *maximum_chunk_size = 24;
  if (chunk_size_information) {
    chunk_size_information->clear();
    chunk_size_information->push_back("Available values [12-24]");
    chunk_size_information->push_back("[compression/decompression]");
  }
  return true;
}

NvcompZstdLibrary::NvcompZstdLibrary() {
  nvcomp_ = new NvcompTemplate(nvcompBatchedZstdCompressGetTempSize,
                               nvcompBatchedZstdCompressGetMaxOutputChunkSize,
                               nvcompBatchedZstdDecompressGetTempSize,
                               nvcompBatchedZstdGetDecompressSizeAsync,
                               nvcompBatchedZstdCompressAsync,
                               nvcompBatchedZstdDecompressAsync);
}

NvcompZstdLibrary::~NvcompZstdLibrary() { delete nvcomp_; }
