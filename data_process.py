# 画像を読み込んで、行列に変換する関数を定義
from keras.preprocessing.image import load_img, img_to_array
def img_to_traindata(file, img_rows, img_cols, rgb):
    if rgb == 0:
        img = load_img(file, color_mode = "grayscale", target_size=(img_rows,img_cols)) # grayscaleで読み込み
    else:
        img = load_img(file, color_mode = "rgb", target_size=(img_rows,img_cols)) # RGBで読み込み
    x = img_to_array(img)
    x = x.astype('float32')
    x /= 255
    return x

# 学習データ、テストデータ生成
import glob, os

img_rows = 224 #　画像サイズはVGG16のデフォルトサイズとする
img_cols = 224
nb_classes = 2 # 怒っている、笑っているの2クラス
img_dirs = ["./dog_angry", "./dog_smile"] # 怒っている犬、笑っている犬の画像を格納したディレクトリ

X_train = []
Y_train = []
X_test = []
Y_test = []
for n, img_dir in enumerate(img_dirs):
    img_files = glob.glob(img_dir+"/*.jpg")   # ディレクトリ内の画像ファイルを全部読み込む
    for i, img_file in enumerate(img_files):  # ディレクトリ(文字種)内の全ファイルに対して
        x = img_to_traindata(img_file, img_rows, img_cols, 1) # 各画像ファイルをRGBで読み込んで行列に変換
        if i < 8: # 1～100枚目までを学習データ
            X_train.append(x) # 学習用データ(入力)に画像を変換した行列を追加
            Y_train.append(n) # 学習用データ(出力)にクラス(怒=0、笑=1)を追加
        else:       # 101～120枚目までをテストデータ
            X_test.append(x) # テストデータ(入力)に画像を変換した行列を追加
            Y_test.append(n) # テストデータ(出力)にクラス(怒=0、笑=1)を追加

import numpy as np
# 学習、テストデータをlistからnumpy.ndarrayに変換
X_train = np.array(X_train, dtype='float') 
Y_train = np.array(Y_train, dtype='int')
X_test = np.array(X_test, dtype='float')
Y_test = np.array(Y_test, dtype='int')

# カテゴリカルデータ(ベクトル)に変換
from keras.utils import np_utils
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

# 作成した学習データ、テストデータをファイル保存
np.save('models/X_train_2class_120.npy', X_train)
np.save('models/X_test_2class_120.npy', X_test)
np.save('models/Y_train_2class_120.npy', Y_train)
np.save('models/Y_test_2class_120.npy', Y_test)

# 作成したデータの型を表示
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# 【パラメータ設定】
batch_size = 20
epochs = 30

input_shape = (img_rows, img_cols, 3)
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# 【モデル定義】
model = Sequential()
model.add(Conv2D(nb_filters, kernel_size, # 畳み込み層
                        padding='valid',
                        activation='relu',
                        input_shape=input_shape))
model.add(Conv2D(nb_filters, kernel_size, activation='relu')) # 畳み込み層
model.add(MaxPooling2D(pool_size=pool_size)) # プーリング層
model.add(Conv2D(nb_filters, kernel_size, activation='relu')) # 畳み込み層
model.add(MaxPooling2D(pool_size=pool_size)) # プーリング層
model.add(Dropout(0.25)) # ドロップアウト(過学習防止のため、入力と出力の間をランダムに切断)

model.add(Flatten()) # 多次元配列を1次元配列に変換
model.add(Dense(128, activation='relu'))  # 全結合層
model.add(Dropout(0.2))  # ドロップアウト
model.add(Dense(nb_classes, activation='sigmoid'))  # 2クラスなので全結合層をsigmoid

# モデルのコンパイル
model.compile(loss='binary_crossentropy', # 2クラスなのでbinary_crossentropy
              optimizer='adam', # 最適化関数のパラメータはデフォルトを使う
              metrics=['accuracy'])

# 【各エポックごとの学習結果を生成するためのコールバックを定義(前回より精度が良い時だけ保存)】
from keras.callbacks import ModelCheckpoint
import os
model_checkpoint = ModelCheckpoint(
    filepath=os.path.join('models','model_2class120_{epoch:02d}.h5'),
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)
print("filepath",os.path.join('models','model_.h5'))

# 【学習】
result = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test),
                   callbacks=[model_checkpoint],validation_split=0.1)
