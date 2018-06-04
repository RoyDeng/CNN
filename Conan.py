import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from sklearn.model_selection import train_test_split

conan_dir = "./data/conan/" # 有柯南圖片的目錄
miku_dir = "./data/not-conan/" # 不是柯南圖片的目錄

images = [] # 用於裝圖片資訊的List
labels = [] # 用於裝標籤的 List

def load_image(path):
    files = os.listdir(path)
    for f in files:
        img_path = path + f
        img_array = img_to_array(load_img(img_path))
        images.append(img_array)
        if "not" in path:
            labels.append(1)
        else:
            labels.append(0)

load_image(conan_dir)
load_image(miku_dir)

data = np.array(images) # 將資料轉為 numpy.ndarray
data /= 255 # 將資料縮減至 0~1 之間

labels = np.array(labels)
labels = np_utils.to_categorical(labels, 2)

# 使用 sklearn 套件中的切資料集模組
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# 建立keras的Sequential模型
model = Sequential()

# 建立第一個卷積層
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', input_shape=(100,100,3), activation='relu'))

# 建立第一個池化層
model.add(MaxPooling2D(pool_size=(2,2)))

# 建立第二個卷積層
model.add(Conv2D(filters=36, kernel_size=(5,5), padding='same', activation='relu'))

# 建立第二個池化層
model.add(MaxPooling2D(pool_size=(2,2)))

# 加入Droput避免overfitting
model.add(Dropout(0.25))

# 建立全鏈結層
model.add(Flatten()) #建立平坦層
model.add(Dense(128, activation='relu')) #建立隱藏層
model.add(Dropout(0.5)) #Dropout
model.add(Dense(2, activation='softmax')) #建立輸出層

# 查看設定的模型
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

train_history=model.fit(x=x_train, y=y_train, epochs=20, verbose=1, validation_data=(x_test,y_test))

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('rain')
    plt.xlabel('poch')
    plt.legend(['train', 'validation'], loc='center right')
    plt.show()

show_train_history(train_history, 'acc','val_acc') #畫出accuracy的執行結果
show_train_history(train_history, 'loss','val_loss') #畫出loss誤差的執行結果
