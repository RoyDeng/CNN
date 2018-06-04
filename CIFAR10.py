# 載入資料集
import numpy as np
np.random.seed(10)
from keras.datasets import cifar10
(x_train_image, y_train_label), (x_test_image, y_test_label)=cifar10.load_data()

# 顯示資料集的結構
x_train_image.shape
x_test_image.shape
y_train_label.shape
y_test_label.shape

# 資料預處理
# 數字圖片
x_train_normalize=x_train_image.astype('float32')/255.0
x_test_normalize=x_test_image.astype('float32')/255.0

# 標籤
from keras.utils import np_utils
y_train_onehot=np_utils.to_categorical(y_train_label)
y_test_onehot=np_utils.to_categorical(y_test_label)

# 建立模型
# 匯入相關模組
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.layers import ZeroPadding2D,Activation

# 建立線性堆疊模型 : 卷積層 (丟棄 25% 神經元) * 2 + 池化層
model=Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(32,32,3), activation='relu'))

model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

# 建立分類模型 (MLP) : 平坦層 + 隱藏層 (1024 神經元) + 輸出層 (10 神經元)
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))

# 檢視模型摘要
print(model.summary())

# 編譯與訓練模型
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_train_normalize, y=y_train_onehot, validation_split=0.2, epochs=10, batch_size=128,verbose=2)

# 繪製訓練結果
def show_train_history(train_history):
    fig=plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(train_history.history["acc"])
    plt.plot(train_history.history["val_acc"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(train_history.history["loss"])
    plt.plot(train_history.history["val_loss"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

import matplotlib.pyplot as plt
show_train_history(train_history)

# 評估預測準確率
scores=model.evaluate(x_test_normalize, y_test_onehot)
print("Accuracy=", scores)
print("Accuracy=", scores[1])

# 預測測試集圖片
prediction=model.predict_classes(x_test_normalize)
print(prediction)
print(prediction[:10])
print(y_test_label[:10])

# 製作混淆矩陣
import pandas as pd
pd.crosstab(y_test_label.reshape(-1), prediction, rownames=['label'],colnames=['predict'])
