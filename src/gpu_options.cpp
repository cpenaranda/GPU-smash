/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

// GPU-SMASH LIBRARIES
#include <gpu_options.hpp>

void GpuOptions::SetCompressionLevel(const uint8_t &compression_level) {
  compression_level_ = compression_level;
  compression_level_set_ = true;
}

void GpuOptions::SetWindowSize(const uint32_t &window_size) {
  window_size_ = window_size;
  window_size_set_ = true;
}

void GpuOptions::SetMode(const uint8_t &mode) {
  mode_ = mode;
  mode_set_ = true;
}

void GpuOptions::SetWorkFactor(const uint8_t &work_factor) {
  work_factor_ = work_factor;
  work_factor_set_ = true;
}

void GpuOptions::SetFlags(const uint8_t &flags) {
  flags_ = flags;
  flags_set_ = true;
}

void GpuOptions::SetChunkSize(const uint8_t &chunk_size) {
  chunk_size_ = chunk_size;
  chunk_size_set_ = true;
}

void GpuOptions::SetChunkNumber(const uint8_t &chunk_number) {
  chunk_number_ = chunk_number;
  chunk_number_set_ = true;
}

void GpuOptions::SetStreamNumber(const uint8_t &stream_number) {
  stream_number_ = stream_number;
  stream_number_set_ = true;
}

void GpuOptions::SetBackReference(const uint8_t &back_reference) {
  back_reference_ = back_reference;
  back_reference_set_ = true;
}

bool GpuOptions::CompressionLevelIsSet() const {
  return compression_level_set_;
}

bool GpuOptions::WindowSizeIsSet() const { return window_size_set_; }

bool GpuOptions::ModeIsSet() const { return mode_set_; }

bool GpuOptions::WorkFactorIsSet() const { return work_factor_set_; }

bool GpuOptions::FlagsIsSet() const { return flags_set_; }

bool GpuOptions::ChunkSizeIsSet() const { return chunk_size_set_; }

bool GpuOptions::ChunkNumberIsSet() const { return chunk_number_set_; }

bool GpuOptions::StreamNumberIsSet() const { return stream_number_set_; }

bool GpuOptions::BackReferenceIsSet() const { return back_reference_set_; }

uint8_t GpuOptions::GetCompressionLevel() const { return compression_level_; }

uint32_t GpuOptions::GetWindowSize() const { return window_size_; }

uint8_t GpuOptions::GetMode() const { return mode_; }

uint8_t GpuOptions::GetWorkFactor() const { return work_factor_; }

uint8_t GpuOptions::GetFlags() const { return flags_; }

uint8_t GpuOptions::GetChunkSize() const { return chunk_size_; }

uint8_t GpuOptions::GetChunkNumber() const { return chunk_number_; }

uint8_t GpuOptions::GetStreamNumber() const { return stream_number_; }

uint8_t GpuOptions::GetBackReference() const { return back_reference_; }

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
  chunk_number_ = 0;
  chunk_number_set_ = false;
  stream_number_ = 0;
  stream_number_set_ = false;
  back_reference_ = 0;
  back_reference_set_ = false;
}

GpuOptions::~GpuOptions() {}
