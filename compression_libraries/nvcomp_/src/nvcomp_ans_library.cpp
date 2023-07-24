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
  bool result =
      GpuCompressionLibrary::CheckStreamNumber("nvcomp-ans", options, 1, 8);
  if (result) {
    configuration_ = {nvcomp_rANS};
    result = nvcomp_->CreateInternalStructures(1, options->GetStreamNumber());
  }
  return result;
}

void NvcompAnsLibrary::GetCompressedDataSize(
    const uint64_t &uncompressed_data_size, uint64_t *compressed_data_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->InitializeCompression(uncompressed_data_size, configuration_,
                                   stream_);
    nvcomp_->GetCompressedDataSize(uncompressed_data_size,
                                   compressed_data_size);
  } else {
    *compressed_data_size = 0;
  }
}

void NvcompAnsLibrary::GetDecompressedDataSizeFromDeviceMemory(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, uint64_t *decompressed_data_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetDecompressedDataSize(device_compressed_data,
                                     decompressed_data_size);
  } else {
    *decompressed_data_size = 0;
  }
}

bool NvcompAnsLibrary::CompressDeviceToDevice(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *device_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressDeviceToDevice(
        device_uncompressed_data, uncompressed_data_size,
        device_compressed_data, compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-ans error when compress data" << std::endl;
    }
  }
  return result;
}

bool NvcompAnsLibrary::CompressHostToDevice(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *device_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressHostToDevice(
        host_uncompressed_data, uncompressed_data_size, device_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-ans error when compress data" << std::endl;
    }
  }
  return result;
}

bool NvcompAnsLibrary::CompressDeviceToHost(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressDeviceToHost(
        device_uncompressed_data, uncompressed_data_size, host_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-ans error when compress data" << std::endl;
    }
  }
  return result;
}

bool NvcompAnsLibrary::CompressHostToHost(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressHostToHost(
        host_uncompressed_data, uncompressed_data_size, host_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-ans error when compress data" << std::endl;
    }
  }
  return result;
}

bool NvcompAnsLibrary::DecompressDeviceToDevice(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressDeviceToDevice(
        device_compressed_data, compressed_data_size, device_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-ans error when decompress data" << std::endl;
    }
  }
  return result;
}

bool NvcompAnsLibrary::DecompressDeviceToHost(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressDeviceToHost(
        device_compressed_data, compressed_data_size, host_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-ans error when decompress data" << std::endl;
    }
  }
  return result;
}

bool NvcompAnsLibrary::DecompressHostToDevice(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressHostToDevice(
        host_compressed_data, compressed_data_size, device_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-ans error when decompress data" << std::endl;
    }
  }
  return result;
}

bool NvcompAnsLibrary::DecompressHostToHost(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressHostToHost(
        host_compressed_data, compressed_data_size, host_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-ans error when decompress data" << std::endl;
    }
  }
  return result;
}

void NvcompAnsLibrary::GetTitle() {
  GpuCompressionLibrary::GetTitle(
      "nvcomp-ans",
      "Proprietary entropy encoder based on asymmetric numeral systems.");
}

bool NvcompAnsLibrary::GetStreamNumberInformation(
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

NvcompAnsLibrary::NvcompAnsLibrary() {
  nvcomp_ = new NvcompTemplate<nvcompBatchedANSOpts_t>(
      nvcompBatchedANSCompressGetTempSize,
      nvcompBatchedANSCompressGetMaxOutputChunkSize,
      nvcompBatchedANSDecompressGetTempSize,
      nvcompBatchedANSGetDecompressSizeAsync, nvcompBatchedANSCompressAsync,
      nvcompBatchedANSDecompressAsync);
}

NvcompAnsLibrary::~NvcompAnsLibrary() { delete nvcomp_; }
