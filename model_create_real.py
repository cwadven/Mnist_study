#직접 데이터셋을 가져와서 리스트 np화 시킨후 문제 정답 --> 모델화
import tensorflow as tf
import numpy as np
from struct import *

x_list = []
y_list = []

x_train = open('train-images.idx3-ubyte','rb') #문제 즉 이미지
y_train =  open('train-labels.idx1-ubyte','rb') #정답

img = np.zeros((28,28)) #이미지가 저장될 부분

s = x_train.read(16)    #read first 16byte  처음에 이거 읽어야 위치를 조정이 된다
l = y_train.read(8)     #read first  8byte  처음에 이거 읽어야 위치를 조정이 된다

k = 0

while True:
    s = x_train.read(784) #바이트 단위로 보임 읽고 그다음 또 읽고 그다음 읽는 방식
    l = y_train.read(1) #바이트 단위로 보임

    if not s:
        break
    if not l:
        break

    index = int(l[0])
    print(k,":",index)

    #unpack(형식, 변수(바이트형의값)) ==> 글자수가 같아야된다 즉 s는 총 784개 그래서 B도 784로
    img = np.reshape(unpack(len(s)*'B',s), (28,28)) #B(unsigned char 784개를 전부 unpack 한다 바이트 들이 문자열로) 그러고 2차원 배열로 만듦
    k=k+1

    x_list.append(img)
    y_list.append(index)

x_list = np.array(x_list)
y_list = np.array(y_list)


# 2. 데이터 전처리 
# 0~255.0 사이의 값을 갖는 픽셀값들을 0~1.0 사이의 값을 갖도록 변환합니다.
# numpy array 여서 나누면 전부다 이렇게 됨! 0~255 값은 진한정도인데 그 진한정도를 0~1로 바꿈 비률화
x_list = x_list/255.0

# 3. 모델 구성
model = tf.keras.models.Sequential([ 
  tf.keras.layers.Flatten(input_shape=(28, 28)), #2차원 배열을 1차원 배열화!
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

 # 4. 모델 컴파일
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# 5. 모델 훈련
model.fit(x_list, y_list, epochs=30)

# 5. 모델 저장
model.save("model.h5")

