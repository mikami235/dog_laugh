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
    filepath=os.path.join('models','model_2class120_{epoch:02d}_{val_acc:.3f}.h5'),
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    verbose=1)

# 【学習】
result = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test),
                   callbacks=[model_checkpoint])
