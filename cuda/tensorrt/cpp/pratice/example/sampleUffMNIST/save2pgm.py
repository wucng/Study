import tensorflow as tf
import os
import PIL.Image

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

labels=[]
for x,y in zip(x_test,y_test):
    if y not in labels:
        PIL.Image.fromarray(x).save("models/%s.pgm"%(y))
        labels.append(y)

    if len(labels)>=10:break