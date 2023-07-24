/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#include <nvcomp/bitcomp.h>

// SMASH LIBRARIES
#include <gpu_options.hpp>
#include <nvcomp_bitcomp_library.hpp>

bool NvcompBitcompLibrary::CheckOptions(GpuOptions *options,
                                        const bool &compressor) {
  bool result =
      GpuCompressionLibrary::CheckFlags("nvcomp-bitcomp", options, 0, 8);
  if ((result = GpuCompressionLibrary::CheckCompressionLevel("nvcomp-bitcomp",
                                                             options, 0, 1))) {
    if ((result = GpuCompressionLibrary::CheckStreamNumber("nvcomp-bitcomp",
                                                           options, 1, 8))) {
      configuration_ = {options->GetCompressionLevel(),
                        static_cast<nvcompType_t>((options->GetFlags() == 8)
                                                      ? 0xff
                                                      : options->GetFlags())};
      result = nvcomp_->CreateInternalStructures(1, options->GetStreamNumber());
    }
  }
  return result;
}

void NvcompBitcompLibrary::GetCompressedDataSize(
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

void NvcompBitcompLibrary::GetDecompressedDataSizeFromDeviceMemory(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, uint64_t *decompressed_data_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    nvcomp_->GetDecompressedDataSize(device_compressed_data,
                                     decompressed_data_size);
  } else {
    *decompressed_data_size = 0;
  }
}

bool NvcompBitcompLibrary::CompressDeviceToDevice(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *device_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressDeviceToDevice(
        device_uncompressed_data, uncompressed_data_size,
        device_compressed_data, compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-bitcomp error when compress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompBitcompLibrary::CompressHostToDevice(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *device_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressHostToDevice(
        host_uncompressed_data, uncompressed_data_size, device_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-bitcomp error when compress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompBitcompLibrary::CompressDeviceToHost(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressDeviceToHost(
        device_uncompressed_data, uncompressed_data_size, host_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-bitcomp error when compress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompBitcompLibrary::CompressHostToHost(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    result = nvcomp_->CompressHostToHost(
        host_uncompressed_data, uncompressed_data_size, host_compressed_data,
        compressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-bitcomp error when compress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompBitcompLibrary::DecompressDeviceToDevice(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressDeviceToDevice(
        device_compressed_data, compressed_data_size, device_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-bitcomp error when decompress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompBitcompLibrary::DecompressDeviceToHost(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressDeviceToHost(
        device_compressed_data, compressed_data_size, host_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-bitcomp error when decompress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompBitcompLibrary::DecompressHostToDevice(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressHostToDevice(
        host_compressed_data, compressed_data_size, device_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-bitcomp error when decompress data"
                << std::endl;
    }
  }
  return result;
}

bool NvcompBitcompLibrary::DecompressHostToHost(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    result = nvcomp_->DecompressHostToHost(
        host_compressed_data, compressed_data_size, host_decompressed_data,
        decompressed_data_size);
    if (!result) {
      std::cout << "ERROR: nvcomp-bitcomp error when decompress data"
                << std::endl;
    }
  }
  return result;
}

void NvcompBitcompLibrary::GetTitle() {
  GpuCompressionLibrary::GetTitle(
      "nvcomp-bitcomp",
      "Proprietary compressor designed for efficient GPU compression in "
      "Scientific Computing applications.");
}

bool NvcompBitcompLibrary::GetCompressionLevelInformation(
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

bool NvcompBitcompLibrary::GetFlagsInformation(
    std::vector<std::string> *flags_information, uint8_t *minimum_flags,
    uint8_t *maximum_flags) {
  if (minimum_flags) *minimum_flags = 0;
  if (maximum_flags) *maximum_flags = 8;
  if (flags_information) {
    flags_information->clear();
    flags_information->push_back("Available values [0-8]");
    flags_information->push_back("0: " + flags_[0]);
    flags_information->push_back("1: " + flags_[1]);
    flags_information->push_back("2: " + flags_[2]);
    flags_information->push_back("3: " + flags_[3]);
    flags_information->push_back("4: " + flags_[4]);
    flags_information->push_back("5: " + flags_[5]);
    flags_information->push_back("6: " + flags_[6]);
    flags_information->push_back("7: " + flags_[7]);
    flags_information->push_back("8: " + flags_[8]);
    flags_information->push_back("[compression]");
  }
  return true;
}

bool NvcompBitcompLibrary::GetStreamNumberInformation(
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

std::string NvcompBitcompLibrary::GetFlagsName(const uint8_t &flags) {
  std::string result = "ERROR";
  if (flags < number_of_flags_) {
    result = flags_[flags];
  }
  return result;
}

NvcompBitcompLibrary::NvcompBitcompLibrary() {
  number_of_flags_ = 9;
  flags_ = new std::string[number_of_flags_];
  flags_[0] = "Char";
  flags_[1] = "Unsigned Char";
  flags_[2] = "Short";
  flags_[3] = "Unsigned Short";
  flags_[4] = "Int";
  flags_[5] = "Unsigned Int";
  flags_[6] = "Long Long";
  flags_[7] = "Unsigned Long Long";
  flags_[8] = "Bits";

  nvcomp_ = new NvcompTemplate<nvcompBatchedBitcompFormatOpts>(
      nvcompBatchedBitcompCompressGetTempSize,
      nvcompBatchedBitcompCompressGetMaxOutputChunkSize,
      nvcompBatchedBitcompDecompressGetTempSize,
      nvcompBatchedBitcompGetDecompressSizeAsync,
      nvcompBatchedBitcompCompressAsync, nvcompBatchedBitcompDecompressAsync);
}

NvcompBitcompLibrary::~NvcompBitcompLibrary() {
  delete[] flags_;
  delete nvcomp_;
}
