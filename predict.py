from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

model = load_model('models/model_2class120_04.h5')
model.summary()

def img_to_traindata(file, img_rows, img_cols, rgb):
    if rgb == 0:
        img = load_img(file, color_mode = "grayscale", target_size=(img_rows,img_cols)) # grayscaleで読み込み
    else:
        img = load_img(file, color_mode = "rgb", target_size=(img_rows,img_cols)) # RGBで読み込み
    x = img_to_array(img)
    x = x.astype('float32')
    x /= 255
    return x

import numpy as np
img_rows = 224 #　画像サイズはVGG16のデフォルトサイズとする
img_cols = 224

## 画像読み込み
filename = "dog_smile/n02085936_37.jpg"
x = img_to_traindata(filename, img_rows, img_cols, 1) # img_to_traindata関数は、学習データ生成のときに定義
x = np.expand_dims(x, axis=0)

## どのクラスかを判別する
preds = model.predict(x)
pred_class = np.argmax(preds[0])
print("識別結果：", pred_class)
print("確率：", preds[0])

from keras import backend as K
import cv2

# モデルの最終出力を取り出す
model_output = model.output[:, pred_class]

# 最後の畳込み層を取り出す
last_conv_output = model.get_layer('conv2d_3').output #'block5_conv3').output

# 最終畳込み層の出力の、モデル最終出力に関しての勾配
grads = K.gradients(model_output, last_conv_output)[0]
# model.inputを入力すると、last_conv_outputとgradsを出力する関数を定義
gradient_function = K.function([model.input], [last_conv_output, grads]) 

# 読み込んだ画像の勾配を求める
output, grads_val = gradient_function([x])
output, grads_val = output[0], grads_val[0]

# 重みを平均化して、レイヤーのアウトプットに乗じてヒートマップ作成
weights = np.mean(grads_val, axis=(0, 1))
heatmap = np.dot(output, weights)

heatmap = cv2.resize(heatmap, (img_rows, img_cols), cv2.INTER_LINEAR)
heatmap = np.maximum(heatmap, 0) 
heatmap = heatmap / heatmap.max()

heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)  # ヒートマップに色をつける
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 色をRGBに変換

# 元の画像と合成
"""
superimposed_img = (np.float32(heatmap)/4 + x[0]*255/4*3)

plt.imshow(superimposed_img)
plt.show()
"""

img = plt.imread(filename, cv2.IMREAD_UNCHANGED)
print(img.shape)  # (330, 440, 4)

fig, ax = plt.subplots()
ax.imshow(img)

plt.show()