import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# config
EPOCHS = 5
MODEL_DIR = "models"
MODEL_VERSION = 1

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


# ## inspect data
# W_grid = 4
# L_grid = 4

# fig, axes = plt.subplots(L_grid, W_grid, figsize = (15, 15))
# axes = axes.ravel()

# n_training = len(X_train)

# for i in np.arange(0, L_grid * W_grid):
#     index = np.random.randint(0, n_training)
#     axes[i].imshow(X_train[index].reshape(28,28))
#     axes[i].set_title(y_train[index])
#     axes[i].axis('off')
    
# plt.subplots_adjust(hspace = 0.4)
# plt.show()


# model building

## first attempt
# classifier = tf.keras.Sequential([
#   tf.keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
#                       strides=2, activation='relu', name='Conv1'),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(10, name='Dense')
# ])

## second attempt
classifier = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation = 'relu'),
  tf.keras.layers.Dense(10, activation = 'softmax')
])

classifier.summary()


classifier.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# model training
classifier.fit(X_train, y_train, epochs=EPOCHS)

test_loss, test_acc = classifier.evaluate(X_test, y_test)
print('\nINFO :: Test accuracy: {}'.format(test_acc))


# save the model
## join the temp model directory with chosen version number
export_path = os.path.join(MODEL_DIR, str(MODEL_VERSION))

## save the model using `save_model`
tf.keras.models.save_model(
    classifier,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

print('\nINFO :: Model Saved!')
