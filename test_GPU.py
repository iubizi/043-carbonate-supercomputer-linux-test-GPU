# -*- coding: UTF-8 -*-

####################
# 避免占满
####################

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

####################
# 加载数据
####################

from tensorflow.keras.datasets import cifar10

# 归一化数据
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 独热编码
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

####################
# 构造模型
####################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout

def get_model():
    
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same', input_shape = (32, 32, 3)))
    model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(512, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))

    return model

####################
# 获取模型
####################

model = get_model()

from tensorflow.keras.optimizers import Adam

model.compile( optimizer = Adam( learning_rate = 1e-4 ),
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'] )

model.summary()

####################
# 训练
####################

model.fit( x_train, y_train,
           validation_data = (x_test, y_test), # 每个epoch冷却一下

           epochs = 20, batch_size = 32,
           verbose = 2, # 2 一次训练就显示一行

           max_queue_size = 1000,
           workers = 8, # 多进程核心数
           use_multiprocessing = True, # 多进程
           )
