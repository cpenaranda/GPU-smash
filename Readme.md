# Smash: Compression abstraction library

Smash is a library that presents a single c++ API to use many compression libraries.

## References
You can access results obtained with the [AI dataset](https://github.com/cpenaranda/AI-dataset) in this [paper](https://doi.org/10.1007/978-3-031-15471-3_21). Also, if you are using this repository in your research, you need to cite the paper:
> Peñaranda, C., Reaño, C., Silla, F. (2022). Smash: A Compression Benchmark with AI Datasets from Remote GPU Virtualization Systems. In: , et al. Hybrid Artificial Intelligent Systems. HAIS 2022. Lecture Notes in Computer Science(), vol 13469. Springer, Cham. https://doi.org/10.1007/978-3-031-15471-3_21

<details><summary>BibTeX</summary>

```
@InProceedings{penaranda2022smash,
  author="Pe{\~{n}}aranda, Cristian and Rea{\~{n}}o, Carlos and Silla, Federico",
  editor="Garc{\'i}a Bringas, Pablo and P{\'e}rez Garc{\'i}a, Hilde and Mart{\'i}nez de Pis{\'o}n, Francisco Javier and Villar Flecha, Jos{\'e} Ram{\'o}n and Troncoso Lora, Alicia and de la Cal, Enrique A. and Herrero, {\'A}lvaro and Mart{\'i}nez {\'A}lvarez, Francisco and Psaila, Giuseppe and Quinti{\'a}n, H{\'e}ctor and Corchado, Emilio",
  title="Smash: A Compression Benchmark with AI Datasets from Remote GPU Virtualization Systems",
  booktitle="Hybrid Artificial Intelligent Systems",
  year="2022",
  publisher="Springer International Publishing",
  address="Cham",
  pages="236--248",
  abstract="Remote GPU virtualization is a mechanism that allows GPU-accelerated applications to be executed in computers without GPUs. Instead, GPUs from remote computers are used. Applications are not aware of using a remote GPU. However, overall performance depends on the throughput of the underlying network connecting the application to the remote GPUs. One way to increase this bandwidth is to compress transmissions made within the remote GPU virtualization middleware between the application side and the GPU side.",
  isbn="978-3-031-15471-3"
}
```

</details>

## How to compile Smash
Smash contains external code. Compression libraries have been added as submodules. Some compression libraries can return an error if the data to compress is too small, or in other situations (e.g. If you try to compress small data, the result could be data that is larger than the original data. That is why some compression libraries prefer to return an error). For that reason, some compression libraries have been modified to remove this behavior and have been stored in local repositories. The idea is the user must decide if compressed data could be used.

An easy way to compile this repository is:

```
git clone git@github.com:cpenaranda/GPU-smash.git
cd GPU-smash
git submodule update --init --force --recursive
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCUDA_TOOLKIT_ROOT_DIR:STRING=/nfs2/LIBS/x86_64/ubuntu20.04/CUDA/11.0/ ..
cmake --build . --config Release --target all
```
