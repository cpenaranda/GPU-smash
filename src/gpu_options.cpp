/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

// SMASH LIBRARIES
#include <gpu_options.hpp>

bool GpuOptions::SetCompressionLevel(const uint8_t &compression_level) {
  compression_level_ = compression_level;
  bool result = compression_level_set_;
  compression_level_set_ = true;
  return result;
}

bool GpuOptions::SetWindowSize(const uint32_t &window_size) {
  window_size_ = window_size;
  bool result = window_size_set_;
  window_size_set_ = true;
  return result;
}

bool GpuOptions::SetMode(const uint8_t &mode) {
  mode_ = mode;
  bool result = mode_set_;
  mode_set_ = true;
  return result;
}

bool GpuOptions::SetWorkFactor(const uint8_t &work_factor) {
  work_factor_ = work_factor;
  bool result = work_factor_set_;
  work_factor_set_ = true;
  return result;
}

bool GpuOptions::SetFlags(const uint8_t &flags) {
  flags_ = flags;
  bool result = flags_set_;
  flags_set_ = true;
  return result;
}

bool GpuOptions::SetChunkSize(const uint8_t &chunk_size) {
  chunk_size_ = chunk_size;
  bool result = chunk_size_set_;
  chunk_size_set_ = true;
  return result;
}

bool GpuOptions::SetBackReferenceBits(const uint8_t &back_reference_bits) {
  back_reference_bits_ = back_reference_bits;
  bool result = back_reference_bits_set_;
  back_reference_bits_set_ = true;
  return result;
}

bool GpuOptions::CompressionLevelIsSet() const {
  return compression_level_set_;
}

bool GpuOptions::WindowSizeIsSet() const { return window_size_set_; }

bool GpuOptions::ModeIsSet() const { return mode_set_; }

bool GpuOptions::WorkFactorIsSet() const { return work_factor_set_; }

bool GpuOptions::FlagsIsSet() const { return flags_set_; }

bool GpuOptions::ChunkSizeIsSet() const { return chunk_size_set_; }

bool GpuOptions::BackReferenceBitsIsSet() const {
  return back_reference_bits_set_;
}

uint8_t GpuOptions::GetCompressionLevel() const { return compression_level_; }

uint32_t GpuOptions::GetWindowSize() const { return window_size_; }

uint8_t GpuOptions::GetMode() const { return mode_; }

uint8_t GpuOptions::GetWorkFactor() const { return work_factor_; }

uint8_t GpuOptions::GetFlags() const { return flags_; }

uint8_t GpuOptions::GetChunkSize() const { return chunk_size_; }

uint8_t GpuOptions::GetBackReferenceBits() const {
  return back_reference_bits_;
}

GpuOptions::GpuOptions() {
  compression_level_ = 0;
  compression_level_set_ = false;
  window_size_ = 0;
  window_size_set_ = false;
  mode_ = 0;
  mode_set_ = false;
  work_factor_ = 0;
  work_factor_set_ = false;
  flags_ = 0;
  flags_set_ = false;
  chunk_size_ = 0;
  chunk_size_set_ = false;
  back_reference_bits_ = 0;
  back_reference_bits_set_ = false;
}

GpuOptions::~GpuOptions() {}
