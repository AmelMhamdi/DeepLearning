import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
from random import randint
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
from tensorflow.contrib.layers import flatten

tf.disable_v2_behavior()

targets = []
features = []

files = glob.glob("cat-and-dog/training_set/training_set/*.jpg")
random.shuffle(files)

for file in files:
    features.append(np.array(Image.open(file).resize((75, 75))))
    target = [1, 0] if "cat" in file else [0, 1]
    targets.append(target)

features = np.array(features)
targets = np.array(targets)

# print("shape features", features.shape)
# print("shape targets", targets.shape)

# Visualisation des images
# for a in (randint(0, len(target)) for _ in range(10)):
# plt.imshow(features[a])
# plt.show()

# Jeu d'entrainement / Jeu de validation
x_train, x_valid, y_train, y_valid = train_test_split(features, targets, test_size=0.1, random_state=42)
print("x_train", x_train.shape)
print("x_valid", x_valid.shape)
print("y_train", y_train.shape)
print("y_valid", y_valid.shape)

# Création du modèle

# Placeholder
x = tf.placeholder(tf.float32, (None, 75, 75, 3), name="Image")
y = tf.placeholder(tf.float32, (None, 2), name="Targets")


def create_conv(prev, filter_size, nb):
    w_filters = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, int(prev.get_shape()[-1]), nb)))
    b_filters = tf.Variable(tf.zeros(shape=(nb)))
    conv = tf.nn.conv2d(prev, w_filters, strides=[1, 1, 1, 1], padding="SAME") + b_filters
    conv =tf.nn.relu(conv)
    conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    return conv


conv = create_conv(x, 8, 32)
conv = create_conv(conv, 5, 64)
conv = create_conv(conv, 5, 64)
conv = create_conv(conv, 5, 128)
conv = create_conv(conv, 5, 256)

flat = flatten(conv)
print(flat)

# Fully connected layer
w1 = tf.Variable(tf.truncated_normal(shape=(int(flat.get_shape()[1]), 521)))
b1 = tf.Variable(tf.zeros(shape=(521)))

fc1 = tf.matmul(flat, w1) + b1
fc1 = tf.nn.relu(fc1)

# Fully connected layer [output]
w2 = tf.Variable(tf.truncated_normal(shape=(521, 2)))
b2 = tf.Variable(tf.zeros(shape=(2)))

logits = tf.matmul(fc1, w2) + b2

softmax = tf.nn.softmax(logits)

# Erreur et Optimisation
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)

# Accuracy
correct_prediction = tf.equal(tf.argmax(softmax, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimizer
optimizer = tf.train.AdamOptimizer(0.001)
train_op = optimizer.minimize(loss_operation)

# Train the model

batch_size = 100

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(0, 1):

      index = np.arange(len(x_train))
      np.random.shuffle(index)
      x_train = x_train[index]
      y_train = y_train[index]

      for b in range(0, len(x_train), batch_size):

          batch = x_train[b:b+batch_size]
          sess.run(train_op, feed_dict={x: batch, y: y_train[b:b+batch_size]})

      accs = []
      for b in range(0, len(x_valid), batch_size):
          batch = x_valid[b:b + batch_size]
          acc = sess.run(accuracy, feed_dict={x: batch, y: y_valid[b:b + batch_size]})
          accs.append(acc)

      print("mean Validation", np.mean(accs))


print("Done")





















