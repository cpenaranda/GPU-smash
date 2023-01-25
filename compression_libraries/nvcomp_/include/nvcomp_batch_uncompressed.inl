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

inline cudaError_t BatchDataUncompressed::GetNextCompression(
    const size_t &chunks, const size_t &chunk_size,
    const cudaStream_t &stream) {
  NvcompUtil::GetNext(data_, d_ptrs_, d_sizes_, chunk_size, chunks, stream);
  data_ += chunks * chunk_size;
  cudaError_t error = cudaSuccess;
  if (size_ < (chunks * chunk_size)) {
    size_ -= ((chunks - 1) * chunk_size);
    error = cudaMemcpyAsync(&d_sizes_[chunks - 1], &size_, sizeof(*d_sizes_),
                            cudaMemcpyHostToDevice, stream);
  } else {
    size_ -= (chunks * chunk_size);
  }
  return error;
}

inline cudaError_t BatchDataUncompressed::InitilizeDecompression(
    char *data, const cudaStream_t &stream) {
  data_ = data;
  begin_data_ = data;
  cudaError_t error =
      cudaMemsetAsync(d_sizes_, 0, sizeof(*d_sizes_) * slices_, stream);
  return cudaSuccess;
}

inline cudaError_t BatchDataUncompressed::GetNextDecompression(
    const size_t &chunks, const size_t &chunk_size,
    const cudaStream_t &stream) {
  NvcompUtil::GetNext(data_, d_ptrs_, d_sizes_, chunk_size, chunks, stream);
  chunk_size_ = chunk_size;
  last_chunk_ = chunks - 1;
  data_ += chunks * chunk_size;
  return cudaSuccess;
}

inline void *const *BatchDataUncompressed::d_ptrs() {
  return reinterpret_cast<void *const *>(d_ptrs_);
}

inline size_t *BatchDataUncompressed::d_sizes() { return d_sizes_; }

inline size_t BatchDataUncompressed::size() {
  size_t last_chunk_size = 0;
  cudaMemcpy(&last_chunk_size, &d_sizes_[last_chunk_], sizeof(last_chunk_size),
             cudaMemcpyDeviceToHost);
  return (data_ - begin_data_) - chunk_size_ + last_chunk_size;
}
