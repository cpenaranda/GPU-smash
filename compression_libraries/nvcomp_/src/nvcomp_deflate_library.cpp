/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#include <nvcomp/deflate.h>

// SMASH LIBRARIES
#include <gpu_options.hpp>
#include <nvcomp_deflate_library.hpp>

bool NvcompDeflateLibrary::CheckOptions(GpuOptions *options,
                                        const bool &compressor) {
  bool result =
      GpuCompressionLibrary::CheckChunkSize("nvcomp-deflate", options, 12, 16);
  if (result) {
    if ((result = GpuCompressionLibrary::CheckCompressionLevel(
             "nvcomp-gdeflate", options, 0, 1))) {
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

void NvcompDeflateLibrary::GetCompressedDataSize(uint64_t uncompressed_size,
                                                 uint64_t *compressed_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetCompressedDataSize(uncompressed_size, compressed_size);
  } else {
    *compressed_size = 0;
  }
}

void NvcompDeflateLibrary::GetDecompressedDataSizeFromDeviceMemory(
    char *device_compressed_data, uint64_t compressed_size,
    uint64_t *decompressed_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetDecompressedDataSize(device_compressed_data, decompressed_size);
  } else {
    *decompressed_size = 0;
  }
}

bool NvcompDeflateLibrary::CompressDeviceMemory(char *device_uncompressed_data,
                                                uint64_t uncompressed_size,
                                                char *device_compressed_data,
                                                uint64_t *compressed_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->Compress(device_uncompressed_data, uncompressed_size,
                               device_compressed_data, compressed_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-deflate error when compress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompDeflateLibrary::DecompressDeviceMemory(
    char *device_compressed_data, uint64_t compressed_size,
    char *device_decompressed_data, uint64_t *decompressed_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->Decompress(device_compressed_data, compressed_size,
                                 device_decompressed_data, decompressed_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-deflate error when decompress data"
                << std::endl;
    }
  }
  return result;
}

void NvcompDeflateLibrary::GetTitle() {
  GpuCompressionLibrary::GetTitle("nvcomp-deflate",
                                  "Huffman + LZ77, Provided for compatibility "
                                  "with existing Deflate-compressed datasets.");
}

bool NvcompDeflateLibrary::GetCompressionLevelInformation(
    std::vector<std::string> *compression_level_information,
    uint8_t *minimum_level, uint8_t *maximum_level) {
  if (minimum_level) *minimum_level = 0;
  if (maximum_level) *maximum_level = 1;
  if (compression_level_information) {
    compression_level_information->clear();
    compression_level_information->push_back("Available values [0-1]");
    compression_level_information->push_back("[compression]");
  }
  return true;
}

bool NvcompDeflateLibrary::GetChunkSizeInformation(
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

NvcompDeflateLibrary::NvcompDeflateLibrary(const uint64_t &batch_size) {
  nvcomp_ =
      new NvcompTemplate(nvcompBatchedDeflateCompressGetTempSize,
                         nvcompBatchedDeflateCompressGetMaxOutputChunkSize,
                         nvcompBatchedDeflateDecompressGetTempSize,
                         nvcompBatchedDeflateGetDecompressSizeAsync,
                         nvcompBatchedDeflateCompressAsync,
                         nvcompBatchedDeflateDecompressAsync, batch_size);
}

NvcompDeflateLibrary::~NvcompDeflateLibrary() { delete nvcomp_; }
