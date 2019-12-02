
import keras 
from keras.datasets import cifar100
from keras.utils import to_categorical
import cv2
import random
import matplotlib.pyplot as plt
from imagenet2str import init

data = init()

model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


(x_train, y_train), (x_test, y_test) = cifar100.load_data() 
average_image = x_train.mean(0)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],x_train.shape[3])



# 9)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


sel_img = random.randint(0,len(x_train))

x_train_resized = cv2.resize(x_train[sel_img], model.input_shape[1:3], interpolation = cv2.INTER_AREA)

plt.figure("Figure " + str(sel_img))
plt.title("Example: " + str(sel_img))
plt.imshow(x_train_resized) 

x_train_resized.resize(1,299,299,3)

erg = model.predict(x_train_resized)

print(data[erg.argmax()])

