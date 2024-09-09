set -e

rm -rf build/*
cd build

CMAKE_CLANG=/usr/lib/llvm-18/cmake
INCL_CLANG=/usr/include/clang/18/include
EXEC_CLANG=/usr/bin/clang++-18

#CMAKE_CLANG=/usr/lib/llvm-17/cmake
#INCL_CLANG=/usr/include/clang/17.0.6/include
#EXEC_CLANG=/usr/bin/clang++-17

#CMAKE_CLANG=/usr/lib/llvm-16/cmake
#INCL_CLANG=/usr/include/clang/16/include
#EXEC_CLANG=/usr/bin/clang++-16

EXEC_NVCXX=/opt/nvidia/hpc_sdk/Linux_x86_64/2024/compilers/bin/nvc++
DIR_CUDA=/opt/nvidia/hpc_sdk/Linux_x86_64/2024/cuda/12.5

cmake -DCMAKE_INSTALL_PREFIX=/home/cbyrohl -DLLVM_DIR=$CMAKE_CLANG -DCLANG_EXECUTABLE_PATH=$EXEC_CLANG -DCLANG_INCLUDE_PATH=$INCL_CLANG -DNVCXX_COMPILER=$EXEC_NVCXX -DCUDA_TOOLKIT_ROOT_DIR=$DIR_CUDA -DWITH_CUDA_BACKEND=ON ..
make -j16
make install
