/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

inline cudaError_t BatchDataUncompressed::InitilizeCompression(
    char *data, const size_t &data_size) {
  data_ = data;
  size_ = data_size;
  return cudaSuccess;
}

inline cudaError_t BatchDataUncompressed::GetNext(const size_t &chunks,
                                                  const size_t &chunk_size,
                                                  const cudaStream_t &stream) {
  for (size_t i_chunk = 0; i_chunk < chunks; ++i_chunk) {
    h_ptrs_[i_chunk] = data_;
    h_sizes_[i_chunk] = chunk_size;
    data_ += chunk_size;
  }
  if (size_ < (chunks * chunk_size)) {
    h_sizes_[chunks - 1] = size_ % chunk_size;
    size_ = 0;
  } else {
    size_ -= (chunks * chunk_size);
  }
  cudaError_t error =
      cudaMemcpyAsync(d_ptrs_, h_ptrs_, chunks * sizeof(*h_ptrs_),
                      cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(d_sizes_, h_sizes_, chunks * sizeof(*h_sizes_),
                            cudaMemcpyHostToDevice, stream);
  }
  return error;
}

inline cudaError_t BatchDataUncompressed::InitilizeDecompression(char *data) {
  data_ = data;
  size_ = 0;
  return cudaSuccess;
}

inline cudaError_t BatchDataUncompressed::GetNext(const size_t &chunks,
                                                  const size_t &chunk_size,
                                                  const cudaStream_t &stream,
                                                  size_t *decompressed_size) {
  for (size_t i_chunk = 0; i_chunk < chunks; ++i_chunk) {
    h_ptrs_[i_chunk] = data_;
    data_ += chunk_size;
  }
  cudaError_t error =
      cudaMemcpyAsync(d_ptrs_, h_ptrs_, chunks * sizeof(*h_ptrs_),
                      cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess) {
    error = cudaMemcpy(h_sizes_, d_sizes_, chunks * sizeof(*h_sizes_),
                       cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) {
      for (size_t i_chunk = 0; i_chunk < chunks; ++i_chunk) {
        *decompressed_size += h_sizes_[i_chunk];
      }
    }
  }
  return error;
}

inline void *const *BatchDataUncompressed::d_ptrs() {
  return reinterpret_cast<void *const *>(d_ptrs_);
}

inline size_t *BatchDataUncompressed::d_sizes() { return d_sizes_; }

inline size_t BatchDataUncompressed::size() { return size_; }
