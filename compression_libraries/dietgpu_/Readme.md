# DietGPU: GPU-based lossless compression for numerical data

## About
[DietGPU](https://github.com/facebookresearch/dietgpu) is a GPU implementation of a fast generalized ANS (asymmetric numeral system) entropy encoder and decoder, with extensions for lossless compression of numerical and other data types in HPC/ML applications.

## Compression Algorithms
* **dietgpu-ans (ANS entropy codec)**
 A generalized byte-oriented range-based ANS (rANS) entropy encoder and decoder.
* **dietgpu-float (Floating point codec)**
 An extension to the above to handle fast lossless compression and decompression of unstructured floating point data, for use in ML and HPC applications, especially in communicating over local interconnects (PCIe / NVLink) and remote interconnects (Ethernet / InfiniBand).

## Options
<details><summary><b><font size="+2">To compress</font></b></summary>
<details><summary><b>dietgpu-ans</b></summary>

  * **Compression level** - (integer, 0-2, default 0)
    * **0** - obtains the fastest compression.
    * **2** - obtains the highest compression ratio.
  * **Chunk size** - (integer, 12-24, default 12)
    * Bits used to indicate the size of slices to compress.
</details>

<details><summary><b>dietgpu-float</b></summary>

  * **Compression level** - (integer, 0-2, default 0)
    * **0** - obtains the fastest compression.
    * **2** - obtains the highest compression ratio.
  * **Chunk size** - (integer, 12-24, default 12)
    * Bits used to indicate the size of slices to compress.
  * **Flags** - (integer, 0-2, default 0)
    * **0 - Float16.**
    * **1 - BFloat16.**
    * **2 - Float32.**
</details>
</details>

<details><summary><b><font size="+2">To decompress</font></b></summary>
<details><summary><b>dietgpu-ans</b></summary>

  * **Compression level** - (integer, 0-2, default 0)
    * **0** - obtains the fastest compression.
    * **2** - obtains the highest compression ratio.
  * **Chunk size** - (integer, 12-24, default 12)
    * Bits used to indicate the size of slices to compress.
</details>

<details><summary><b>dietgpu-float</b></summary>

  * **Compression level** - (integer, 0-2, default 0)
    * **0** - obtains the fastest compression.
    * **2** - obtains the highest compression ratio.
  * **Chunk size** - (integer, 12-24, default 12)
    * Bits used to indicate the size of slices to compress.
  * **Flags** - (integer, 0-2, default 0)
    * **0 - Float16.**
    * **1 - BFloat16.**
    * **2 - Float32.**
</details>
</details>

## License
DietGPU is licensed with the [MIT License](https://github.com/facebookresearch/dietgpu/blob/main/LICENSE.md).