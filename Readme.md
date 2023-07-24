# GPU-Smash: Compression abstraction library

GPU-Smash library presents a single C++ API to use many GPU-based compression libraries.

## References
In process..

</details>

## How to compile GPU-Smash
GPU-Smash contains external code. Compression libraries have been added as submodules. Some compression libraries can return an error if the data to compress is too small or in other situations (e.g., if you try to compress small data, the result could be data that is larger than the original data. That is why some compression libraries prefer to return an error). Therefore, some compression libraries have been modified to remove this behavior and stored in local repositories. The idea is the user must decide if compressed data could be used.

An easy way to compile this repository is as follows:

```
git clone git@github.com:cpenaranda/GPU-smash.git
cd GPU-smash
git submodule update --init --force --recursive
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCUDA_TOOLKIT_ROOT_DIR:STRING=/PATH_TO_CUDA_CUDA/11.0/ ..
cmake --build . --config Release --target all
```

## How to run GPU-Smash
In process..

## Different options available
In process..

## Libraries used in GPU-Smash
|     |     |     | Name |     |     |     |
| :-: | :-: | :-: | :--: | :-: | :-: | :-: |
|     |     |     |      |     |     |     |
