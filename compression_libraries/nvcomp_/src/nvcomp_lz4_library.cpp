/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#include <nvcomp/lz4.h>

// SMASH LIBRARIES
#include <gpu_options.hpp>
#include <nvcomp_lz4_library.hpp>

bool NvcompLz4Library::CheckOptions(GpuOptions *options,
                                    const bool &compressor) {
  bool result =
      GpuCompressionLibrary::CheckChunkSize("nvcomp-lz4", options, 12, 24);
  if (result) {
    if ((result =
             GpuCompressionLibrary::CheckFlags("nvcomp-lz4", options, 0, 6))) {
      if ((result = GpuCompressionLibrary::CheckChunkNumber("nvcomp-lz4",
                                                            options, 4, 13))) {
        if ((result = GpuCompressionLibrary::CheckStreamNumber(
                 "nvcomp-lz4", options, 1, 8))) {
          if ((result = nvcomp_->CreateInternalStructures(
                   options->GetChunkNumber(), options->GetStreamNumber()))) {
            chunk_size_ = 1 << options->GetChunkSize();
            configuration_ = {static_cast<nvcompType_t>(
                (options->GetFlags() == 6) ? 0xff : options->GetFlags())};
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

void NvcompLz4Library::GetCompressedDataSize(
    const uint64_t &uncompressed_data_size, uint64_t *compressed_data_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetCompressedDataSize(uncompressed_data_size,
                                   compressed_data_size);
  } else {
    *compressed_data_size = 0;
  }
}

void NvcompLz4Library::GetDecompressedDataSizeFromDeviceMemory(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, uint64_t *decompressed_data_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetDecompressedDataSize(device_compressed_data,
                                     decompressed_data_size);
  } else {
    *decompressed_data_size = 0;
  }
}

bool NvcompLz4Library::CompressDeviceToDevice(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *device_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressDeviceToDevice(
        device_uncompressed_data, uncompressed_data_size,
        device_compressed_data, compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-lz4 error when compress data" << std::endl;
    }
  }
  return result;
}

bool NvcompLz4Library::CompressHostToDevice(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *device_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressHostToDevice(
        host_uncompressed_data, uncompressed_data_size, device_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-lz4 error when compress data" << std::endl;
    }
  }
  return result;
}

bool NvcompLz4Library::CompressDeviceToHost(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressDeviceToHost(
        device_uncompressed_data, uncompressed_data_size, host_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-lz4 error when compress data" << std::endl;
    }
  }
  return result;
}

bool NvcompLz4Library::CompressHostToHost(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressHostToHost(
        host_uncompressed_data, uncompressed_data_size, host_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-lz4 error when compress data" << std::endl;
    }
  }
  return result;
}

bool NvcompLz4Library::DecompressDeviceToDevice(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressDeviceToDevice(
        device_compressed_data, compressed_data_size, device_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-lz4 error when decompress data" << std::endl;
    }
  }
  return result;
}

bool NvcompLz4Library::DecompressDeviceToHost(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressDeviceToHost(
        device_compressed_data, compressed_data_size, host_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-lz4 error when decompress data" << std::endl;
    }
  }
  return result;
}

bool NvcompLz4Library::DecompressHostToDevice(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressHostToDevice(
        host_compressed_data, compressed_data_size, device_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-lz4 error when decompress data" << std::endl;
    }
  }
  return result;
}

bool NvcompLz4Library::DecompressHostToHost(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressHostToHost(
        host_compressed_data, compressed_data_size, host_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-lz4 error when decompress data" << std::endl;
    }
  }
  return result;
}

void NvcompLz4Library::GetTitle() {
  GpuCompressionLibrary::GetTitle(
      "nvcomp-lz4",
      "General-purpose no-entropy byte-level compressor well-suited for a wide "
      "range of datasets.");
}

bool NvcompLz4Library::GetFlagsInformation(
    std::vector<std::string> *flags_information, uint8_t *minimum_flags,
    uint8_t *maximum_flags) {
  if (minimum_flags) *minimum_flags = 0;
  if (maximum_flags) *maximum_flags = 6;
  if (flags_information) {
    flags_information->clear();
    flags_information->push_back("Available values [0-6]");
    flags_information->push_back("0: " + flags_[0]);
    flags_information->push_back("1: " + flags_[1]);
    flags_information->push_back("2: " + flags_[2]);
    flags_information->push_back("3: " + flags_[3]);
    flags_information->push_back("4: " + flags_[4]);
    flags_information->push_back("5: " + flags_[5]);
    flags_information->push_back("6: " + flags_[6]);
    flags_information->push_back("[compression]");
  }
  return true;
}

bool NvcompLz4Library::GetChunkSizeInformation(
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

bool NvcompLz4Library::GetChunkNumberInformation(
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

bool NvcompLz4Library::GetStreamNumberInformation(
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

std::string NvcompLz4Library::GetFlagsName(const uint8_t &flags) {
  std::string result = "ERROR";
  if (flags < number_of_flags_) {
    result = flags_[flags];
  }
  return result;
}

NvcompLz4Library::NvcompLz4Library() {
  number_of_flags_ = 7;
  flags_ = new std::string[number_of_flags_];
  flags_[0] = "Char";
  flags_[1] = "Unsigned Char";
  flags_[2] = "Short";
  flags_[3] = "Unsigned Short";
  flags_[4] = "Int";
  flags_[5] = "Unsigned Int";
  flags_[6] = "Bits";

  nvcomp_ = new NvcompTemplate<nvcompBatchedLZ4Opts_t>(
      nvcompBatchedLZ4CompressGetTempSize,
      nvcompBatchedLZ4CompressGetMaxOutputChunkSize,
      nvcompBatchedLZ4DecompressGetTempSize,
      nvcompBatchedLZ4GetDecompressSizeAsync, nvcompBatchedLZ4CompressAsync,
      nvcompBatchedLZ4DecompressAsync);
}

NvcompLz4Library::~NvcompLz4Library() {
  delete[] flags_;
  delete nvcomp_;
}
