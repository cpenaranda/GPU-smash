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
      GpuCompressionLibrary::CheckChunkSize("nvcomp-zstd", options, 12, 16);
  if (result) {
    if ((result = GpuCompressionLibrary::CheckChunkNumber("nvcomp-zstd",
                                                          options, 4, 13))) {
      if ((result = GpuCompressionLibrary::CheckStreamNumber("nvcomp-zstd",
                                                             options, 1, 8))) {
        if ((result = nvcomp_->CreateInternalStructures(
                 options->GetChunkNumber(), options->GetStreamNumber()))) {
          chunk_size_ = 1 << options->GetChunkSize();
          if (compressor) {
            configuration_ = {0};
            result = nvcomp_->InitializeCompression(chunk_size_, configuration_,
                                                    stream_);
          } else {
            result = nvcomp_->InitializeDecompression(chunk_size_, stream_);
          }
        }
      }
    }
  }
  return result;
}

void NvcompZstdLibrary::GetCompressedDataSize(
    const uint64_t &uncompressed_data_size, uint64_t *compressed_data_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetCompressedDataSize(uncompressed_data_size,
                                   compressed_data_size);
  } else {
    *compressed_data_size = 0;
  }
}

void NvcompZstdLibrary::GetDecompressedDataSizeFromDeviceMemory(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, uint64_t *decompressed_data_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetDecompressedDataSize(device_compressed_data,
                                     decompressed_data_size);
  } else {
    *decompressed_data_size = 0;
  }
}

bool NvcompZstdLibrary::CompressDeviceToDevice(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *device_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressDeviceToDevice(
        device_uncompressed_data, uncompressed_data_size,
        device_compressed_data, compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-zstd error when compress data" << std::endl;
    }
  }
  return result;
}

bool NvcompZstdLibrary::CompressHostToDevice(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *device_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressHostToDevice(
        host_uncompressed_data, uncompressed_data_size, device_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-zstd error when compress data" << std::endl;
    }
  }
  return result;
}

bool NvcompZstdLibrary::CompressDeviceToHost(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressDeviceToHost(
        device_uncompressed_data, uncompressed_data_size, host_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-zstd error when compress data" << std::endl;
    }
  }
  return result;
}

bool NvcompZstdLibrary::CompressHostToHost(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressHostToHost(
        host_uncompressed_data, uncompressed_data_size, host_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-zstd error when compress data" << std::endl;
    }
  }
  return result;
}

bool NvcompZstdLibrary::DecompressDeviceToDevice(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressDeviceToDevice(
        device_compressed_data, compressed_data_size, device_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-zstd error when decompress data" << std::endl;
    }
  }
  return result;
}

bool NvcompZstdLibrary::DecompressDeviceToHost(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressDeviceToHost(
        device_compressed_data, compressed_data_size, host_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-zstd error when decompress data" << std::endl;
    }
  }
  return result;
}

bool NvcompZstdLibrary::DecompressHostToDevice(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressHostToDevice(
        host_compressed_data, compressed_data_size, device_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-zstd error when decompress data" << std::endl;
    }
  }
  return result;
}

bool NvcompZstdLibrary::DecompressHostToHost(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressHostToHost(
        host_compressed_data, compressed_data_size, host_decompressed_data,
        decompressed_data_size);
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
  if (maximum_chunk_size) *maximum_chunk_size = 16;
  if (chunk_size_information) {
    chunk_size_information->clear();
    chunk_size_information->push_back("Available values [12-16]");
    chunk_size_information->push_back("[compression/decompression]");
  }
  return true;
}

bool NvcompZstdLibrary::GetChunkNumberInformation(
    std::vector<std::string> *chunk_number_information, uint8_t *minimum_chunks,
    uint8_t *maximum_chunks) {
  if (minimum_chunks) *minimum_chunks = 4;
  if (maximum_chunks) *maximum_chunks = 13;
  if (chunk_number_information) {
    chunk_number_information->clear();
    chunk_number_information->push_back("Available values [4-13]");
    chunk_number_information->push_back("[compression]");
  }
  return true;
}

bool NvcompZstdLibrary::GetStreamNumberInformation(
    std::vector<std::string> *stream_number_information,
    uint8_t *minimum_streams, uint8_t *maximum_streams) {
  if (minimum_streams) *minimum_streams = 1;
  if (maximum_streams) *maximum_streams = 8;
  if (stream_number_information) {
    stream_number_information->clear();
    stream_number_information->push_back("Available values [1-8]");
    stream_number_information->push_back("[compression]");
  }
  return true;
}

NvcompZstdLibrary::NvcompZstdLibrary() {
  nvcomp_ = new NvcompTemplate<nvcompBatchedZstdOpts_t>(
      nvcompBatchedZstdCompressGetTempSize,
      nvcompBatchedZstdCompressGetMaxOutputChunkSize,
      nvcompBatchedZstdDecompressGetTempSize,
      nvcompBatchedZstdGetDecompressSizeAsync, nvcompBatchedZstdCompressAsync,
      nvcompBatchedZstdDecompressAsync);
}

NvcompZstdLibrary::~NvcompZstdLibrary() { delete nvcomp_; }
