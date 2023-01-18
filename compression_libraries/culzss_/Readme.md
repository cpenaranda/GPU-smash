# CULZSS: GPU-based LZSS compression algorithm

## About
[CULZSS](https://github.com/adnanozsoy/CUDA_Compression) is a GPU-based LZSS compression algorithm, highly tuned for NVIDIA GPGPUs and for streaming data, leveraging the respective strengths of CPUs and GPUs together.

## Options
### To compress
* **Chunk size** - (integer, 12-24, default 12)
  * Bits used to indicate the size of slices to compress.

### To decompress
* **Chunk size** - (integer, 12-24, default 12)
  * Bits used to indicate the size of slices to compress.

## License
CULZSS is licensed with the [Apache License](http://www.apache.org/licenses/LICENSE-2.0).