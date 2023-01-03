import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import requests

# prepare data
## import Fashion MNIST Dataset using Keras 
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

## data normalization -> Between 0 and 1 
X_train = X_train / 255.0
X_test = X_test / 255.0

## reshape data to be = (no_of_images, 28, 28, 1) instead of (no_of_images, 28,28)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

## define images classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

## display an test image selected below
def show(idx, title):
  plt.figure()
  plt.imshow(X_test[idx].reshape(28,28))
  plt.title('\n\n{}'.format(title), fontdict={'size': 16})
  plt.show()

# ## select a random test image and get it's class label
# random = np.random.randint(0, len(X_test)-1)
# show(random, 'Test Image Class: {}'.format(class_names[y_test[random]]))


# running predictions
## prepare a list of first 3 images to be send to the REST API
data = json.dumps({"signature_name": "serving_default", "instances": X_test[0:3].tolist()})

## send request
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/mnist_fashion:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']


# ## display first result
# show(0, 'Prediction: {} (class {}) / True: {} (class {})'.format(
#   class_names[np.argmax(predictions[0])], y_test[0], class_names[np.argmax(predictions[0])], y_test[0]))

## loop over all results
for i in range(0,3):
  show(i, 'Prediction: {} (class {}) / True: {} (class {})'.format(
    class_names[np.argmax(predictions[i])], y_test[i], class_names[np.argmax(predictions[i])], y_test[i]))