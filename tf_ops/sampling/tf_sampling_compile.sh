#/bin/bash
CUDA_PATH=/usr/local/cuda-9.0
$CUDA_PATH/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $TF_INC -I $CUDA_PATH/include -lcudart -L $CUDA_PATH/lib64/ -L$TF_LIB -I$TF_INC/external/nsync/public -ltensorflow_framework  -O2 -D_GLIBCXX_USE_CXX11_ABI=0
