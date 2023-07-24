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
      if ((result = GpuCompressionLibrary::CheckChunkNumber("nvcomp-gdeflate",
                                                            options, 4, 13))) {
        if ((result = GpuCompressionLibrary::CheckStreamNumber(
                 "nvcomp-gdeflate", options, 1, 8))) {
          if ((result = nvcomp_->CreateInternalStructures(
                   options->GetChunkNumber(), options->GetStreamNumber()))) {
            chunk_size_ = 1 << options->GetChunkSize();
            configuration_ = {options->GetCompressionLevel()};
            if (compressor) {
              result = nvcomp_->InitializeCompression(chunk_size_,
                                                      configuration_, stream_);
            } else {
              result = nvcomp_->InitializeDecompression(chunk_size_, stream_);
            }
          }
        }
      }
    }
  }
  return result;
}

void NvcompGdeflateLibrary::GetCompressedDataSize(
    const uint64_t &uncompressed_data_size, uint64_t *compressed_data_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetCompressedDataSize(uncompressed_data_size,
                                   compressed_data_size);
  } else {
    *compressed_data_size = 0;
  }
}

void NvcompGdeflateLibrary::GetDecompressedDataSizeFromDeviceMemory(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, uint64_t *decompressed_data_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetDecompressedDataSize(device_compressed_data,
                                     decompressed_data_size);
  } else {
    *decompressed_data_size = 0;
  }
}

bool NvcompGdeflateLibrary::CompressDeviceToDevice(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *device_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressDeviceToDevice(
        device_uncompressed_data, uncompressed_data_size,
        device_compressed_data, compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-gdeflate error when compress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompGdeflateLibrary::CompressHostToDevice(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *device_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressHostToDevice(
        host_uncompressed_data, uncompressed_data_size, device_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-gdeflate error when compress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompGdeflateLibrary::CompressDeviceToHost(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressDeviceToHost(
        device_uncompressed_data, uncompressed_data_size, host_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-gdeflate error when compress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompGdeflateLibrary::CompressHostToHost(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressHostToHost(
        host_uncompressed_data, uncompressed_data_size, host_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-gdeflate error when compress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompGdeflateLibrary::DecompressDeviceToDevice(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressDeviceToDevice(
        device_compressed_data, compressed_data_size, device_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-gdeflate error when decompress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompGdeflateLibrary::DecompressDeviceToHost(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressDeviceToHost(
        device_compressed_data, compressed_data_size, host_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-gdeflate error when decompress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompGdeflateLibrary::DecompressHostToDevice(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressHostToDevice(
        host_compressed_data, compressed_data_size, device_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-gdeflate error when decompress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompGdeflateLibrary::DecompressHostToHost(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressHostToHost(
        host_compressed_data, compressed_data_size, host_decompressed_data,
        decompressed_data_size);
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

bool NvcompGdeflateLibrary::GetChunkNumberInformation(
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

bool NvcompGdeflateLibrary::GetStreamNumberInformation(
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

NvcompGdeflateLibrary::NvcompGdeflateLibrary() {
  nvcomp_ = new NvcompTemplate<nvcompBatchedGdeflateOpts_t>(
      nvcompBatchedGdeflateCompressGetTempSize,
      nvcompBatchedGdeflateCompressGetMaxOutputChunkSize,
      nvcompBatchedGdeflateDecompressGetTempSize,
      nvcompBatchedGdeflateGetDecompressSizeAsync,
      nvcompBatchedGdeflateCompressAsync, nvcompBatchedGdeflateDecompressAsync);
}

NvcompGdeflateLibrary::~NvcompGdeflateLibrary() { delete nvcomp_; }
