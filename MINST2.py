import tensorflow as tf
import cv2
import numpy as np
import math
from scipy import ndimage


def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted



#---------------------------------------------------------------------------------------------------------- 모델
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_model = tf.matmul(x, W) + b

#---------------------------------------------------------------------------------------------------------- 라벨
y = tf.placeholder(tf.float32, [None, 10])



#---------------------------------------------------------------------------------------------------------- 세션
sess = tf.Session()

sess.run(tf.global_variables_initializer())



# 저장된 모델 파라미터를 가져옵니다.
model_path = "/tmp/model.saved"
saver = tf.train.Saver()

saver.restore(sess, model_path)
print("Model restored from file: %s" % model_path)




# 10개의 이미지를 위한 배열을 생성
images = np.zeros((10,784))

i = 0
for no in range(10): # 10개의 이미지를 입력 받음


    gray = cv2.imread(str(no)+".png", 0)
    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)



    (thresh, gray) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV  | cv2.THRESH_OTSU)

    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    rows,cols = gray.shape

    cv2.imwrite("b_" + str(no) + ".png", gray)



    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    gray = shifted


    cv2.imwrite("image_" + str(no) + ".png", gray)


    flatten = gray.flatten() / 255.0
    images[i] = flatten

    i += 1



print (sess.run(tf.argmax(y_model,1), feed_dict={x: images}))