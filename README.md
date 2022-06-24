# 043-carbonate-supercomputer-test-GPU

043 carbonate supercomputer test GPU

标准slurm作业脚本

## output

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 64)        1792      
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 64)        36928     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0         
_________________________________________________________________
batch_normalization (BatchNo (None, 16, 16, 64)        256       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 128)       73856     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 128)       147584    
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 128)         0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 8, 8, 128)         512       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 256)         295168    
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 256)         590080    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 256)         0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 4, 4, 256)         1024      
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 4, 4, 512)         1180160   
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 4, 4, 512)         2359808   
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 2, 512)         0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 2, 2, 512)         2048      
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0         
_________________________________________________________________
dense (Dense)                (None, 4096)              8392704   
_________________________________________________________________
dropout (Dropout)            (None, 4096)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              16781312  
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                40970     
=================================================================
Total params: 29,904,202
Trainable params: 29,902,282
Non-trainable params: 1,920
_________________________________________________________________
Epoch 1/20
1563/1563 - 22s - loss: 1.4784 - accuracy: 0.4772 - val_loss: 1.3045 - val_accuracy: 0.5324
Epoch 2/20
1563/1563 - 17s - loss: 0.9766 - accuracy: 0.6542 - val_loss: 0.8841 - val_accuracy: 0.6879
Epoch 3/20
1563/1563 - 17s - loss: 0.7440 - accuracy: 0.7403 - val_loss: 0.9788 - val_accuracy: 0.6620
Epoch 4/20
1563/1563 - 17s - loss: 0.5777 - accuracy: 0.7985 - val_loss: 0.8180 - val_accuracy: 0.7227
Epoch 5/20
1563/1563 - 17s - loss: 0.4439 - accuracy: 0.8458 - val_loss: 0.6731 - val_accuracy: 0.7736
Epoch 6/20
1563/1563 - 17s - loss: 0.3278 - accuracy: 0.8857 - val_loss: 0.7186 - val_accuracy: 0.7708
Epoch 7/20
1563/1563 - 17s - loss: 0.2492 - accuracy: 0.9144 - val_loss: 0.7351 - val_accuracy: 0.7748
Epoch 8/20
1563/1563 - 17s - loss: 0.1984 - accuracy: 0.9319 - val_loss: 0.7995 - val_accuracy: 0.7771
Epoch 9/20
1563/1563 - 17s - loss: 0.1586 - accuracy: 0.9458 - val_loss: 0.9237 - val_accuracy: 0.7632
Epoch 10/20
1563/1563 - 17s - loss: 0.1430 - accuracy: 0.9512 - val_loss: 0.9776 - val_accuracy: 0.7723
Epoch 11/20
1563/1563 - 17s - loss: 0.1252 - accuracy: 0.9577 - val_loss: 0.9406 - val_accuracy: 0.7717
Epoch 12/20
1563/1563 - 17s - loss: 0.1135 - accuracy: 0.9621 - val_loss: 1.0336 - val_accuracy: 0.7739
Epoch 13/20
1563/1563 - 17s - loss: 0.1043 - accuracy: 0.9652 - val_loss: 0.9601 - val_accuracy: 0.7894
Epoch 14/20
1563/1563 - 17s - loss: 0.0970 - accuracy: 0.9679 - val_loss: 0.9782 - val_accuracy: 0.7894
Epoch 15/20
1563/1563 - 17s - loss: 0.0899 - accuracy: 0.9712 - val_loss: 0.9551 - val_accuracy: 0.7946
Epoch 16/20
1563/1563 - 17s - loss: 0.0799 - accuracy: 0.9733 - val_loss: 1.0922 - val_accuracy: 0.7807
Epoch 17/20
1563/1563 - 17s - loss: 0.0803 - accuracy: 0.9739 - val_loss: 1.0328 - val_accuracy: 0.7902
Epoch 18/20
1563/1563 - 17s - loss: 0.0773 - accuracy: 0.9755 - val_loss: 0.9670 - val_accuracy: 0.7897
Epoch 19/20
1563/1563 - 17s - loss: 0.0723 - accuracy: 0.9771 - val_loss: 1.0140 - val_accuracy: 0.7893
Epoch 20/20
1563/1563 - 17s - loss: 0.0709 - accuracy: 0.9775 - val_loss: 1.0790 - val_accuracy: 0.7905
```

## error

```
Python programming language version 3.8.2 loaded.
gcc version 6.3.0 unloaded.
gcc version 9.1.0 loaded.
Python programming language version 3.8.2 unloaded.
openmpi version 4.0.1 loaded.
Deep Learning stack for Cuda 11.2 loaded
2022-06-24 01:29:48.585733: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-06-24 01:29:49.124502: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
2022-06-24 01:29:49.124565: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 15405 MB memory:  -> device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:58:00.0, compute capability: 6.0
2022-06-24 01:29:51.321000: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2022-06-24 01:29:53.276541: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8100
```
