# NVCOMP - High Speed Data Compression Using NVIDIA GPUs

## About
[Nvcomp](https://developer.nvidia.com/nvcomp) provides fast lossless data compression and decompression using a GPU. It features generic compression interfaces to enable developers to use high-performance GPU compressors in their applications.

## Compression Algorithms
* **nvcomp-ans (ANS)**
 Proprietary entropy encoder based on asymmetric numeral systems.
* **nvcomp-bitcomp (Bitcomp)**
 Proprietary compressor designed for efficient GPU compression in Scientific Computing applications.
* **nvcomp-deflate (Deflate)**
 Huffman + LZ77, Provided for compatibility with existing Deflate-compressed datasets.
* **nvcomp-gdeflate (GDeflate)**
 A new compression format that closely matches the DEFLATE format and allows more efficient GPU decompression.
* **nvcomp-lz4 ([LZ4](https://github.com/lz4/lz4))**
 General-purpose no-entropy byte-level compressor well-suited for a wide range of datasets.
* **nvcomp-snappy ([Snappy](https://github.com/google/snappy))**
 Similar to LZ4, this byte-level compressor is a popular existing format used for tabular data.
* **nvcomp-zstd ([zStandard](https://github.com/facebook/zstd))**
 Huffman + LZ77 + ANS, popular compression format developed by Meta.

## Options
<details><summary><b><font size="+2">To compress</font></b></summary>
<details><summary><b>nvcomp-bitcomp</b></summary>

  * **Compression level** - (integer, 0-1, default 0)
    * **0** - obtains the fastest compression.
    * **1** - obtains the highest compression ratio.
  * **Flags** - (integer, 0-8, default 0)
    * **0 - Char.**
    * **1 - Unsigned Char.**
    * **2 - Short.**
    * **3 - Unsigned Short.**
    * **4 - Int.**
    * **5 - Unsigned Int.**
    * **6 - Long Long.**
    * **7 - Unsigned Long Long.**
    * **8 - Bits.**
</details>

<details><summary><b>nvcomp-deflate</b></summary>

  * **Compression level** - (integer, 0-1, default 0)
    * **0** - obtains the fastest compression.
    * **1** - obtains the highest compression ratio.
  * **Chunk size** - (integer, 12-24, default 12)
    * Bits used to indicate the size of slices to compress.
</details>

<details><summary><b>nvcomp-gdeflate</b></summary>

  * **Compression level** - (integer, 0-2, default 0)
    * **0** - obtains the fastest compression.
    * **2** - obtains the highest compression ratio.
  * **Chunk size** - (integer, 12-24, default 12)
    * Bits used to indicate the size of slices to compress.
</details>

<details><summary><b>nvcomp-lz4</b></summary>

  * **Flags** - (integer, 0-6, default 0)
    * **0 - Char.**
    * **1 - Unsigned Char.**
    * **2 - Short.**
    * **3 - Unsigned Short.**
    * **4 - Int.**
    * **5 - Unsigned Int.**
    * **6 - Bits.**
  * **Chunk size** - (integer, 12-24, default 12)
    * Bits used to indicate the size of slices to compress.
</details>

<details><summary><b>nvcomp-snappy</b></summary>

  * **Chunk size** - (integer, 12-24, default 12)
    * Bits used to indicate the size of slices to compress.
</details>

<details><summary><b>nvcomp-zstd</b></summary>

  * **Chunk size** - (integer, 12-24, default 12)
    * Bits used to indicate the size of slices to compress.
</details>
</details>

<details><summary><b><font size="+2">To decompress</font></b></summary>
<details><summary><b>nvcomp-deflate</b></summary>

  * **Chunk size** - (integer, 12-24, default 12)
    * Bits used to indicate the size of slices to compress.
</details>

<details><summary><b>nvcomp-gdeflate</b></summary>

  * **Chunk size** - (integer, 12-24, default 12)
    * Bits used to indicate the size of slices to compress.
</details>

<details><summary><b>nvcomp-lz4</b></summary>

  * **Chunk size** - (integer, 12-24, default 12)
    * Bits used to indicate the size of slices to compress.
</details>

<details><summary><b>nvcomp-snappy</b></summary>

  * **Chunk size** - (integer, 12-24, default 12)
    * Bits used to indicate the size of slices to compress.
</details>

<details><summary><b>nvcomp-zstd</b></summary>

  * **Chunk size** - (integer, 12-24, default 12)
    * Bits used to indicate the size of slices to compress.
</details>
</details>

## License
Nvcomp is licensed with the [NVIDIA Software License Agreement](https://developer.download.nvidia.com/compute/nvcomp/2.3/LICENSE.txt).