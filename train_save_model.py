import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import datetime
import os
import io

# config
EPOCHS = 5
MODEL_DIR = "models"
MODEL_VERSION = 1


############################################################################################
#################################### PREPROCESSING #########################################
############################################################################################

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


############################################################################################
##################################### BUILD MODEL ##########################################
############################################################################################


# model building

# ## first attempt
# classifier = tf.keras.Sequential([
#   tf.keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
#                       strides=2, activation='relu', name='Conv1'),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(10, name='Dense')
# ])

## second attempt (best!)
classifier = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation = 'relu'),
  tf.keras.layers.Dense(10, activation = 'softmax')
])

# ## third attempt
# classifier = tf.keras.Sequential([
#   tf.keras.layers.Conv2D(6, (5,5), activation = 'relu', input_shape = (28,28,1)),
#   tf.keras.layers.AveragePooling2D(),
#   tf.keras.layers.Conv2D(16, (5,5), activation = 'relu'),
#   tf.keras.layers.AveragePooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(120, activation = 'relu'),
#   tf.keras.layers.Dense(84, activation = 'relu'),
#   tf.keras.layers.Dense(10, activation = 'softmax')
# ])


classifier.summary()


classifier.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


############################################################################################
##################################### TENSORBOARD ##########################################
############################################################################################

##################################### PRINT IMAGES #########################################

# configuring tensorboard
## set log data location
log_dir="tensorboard/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
## create a file writer for the log directory
file_writer = tf.summary.create_file_writer(log_dir)

def plot_to_image(figure):
  # Converts the matplotlib plot specified by 'figure' to a PNG image and
  # returns it. The supplied figure is closed and inaccessible after this call.
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def image_grid():
  # Return a 5x5 grid of the MNIST images as a matplotlib figure.
  # Create a figure to contain the plot.
  figure = plt.figure(figsize=(10,10))
  for i in range(25):
    # Start next subplot.
    plt.subplot(5, 5, i + 1, title=class_names[y_train[i]])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)

  return figure

# Prepare the plot
figure = image_grid()
# Convert to image and log
with file_writer.as_default():
  tf.summary.image("Training data", plot_to_image(figure), step=0)

## create Tensorboard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

####################################### PRINT CM ###########################################

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion Matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')


def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred_raw = classifier.predict(X_test)
  test_pred = np.argmax(test_pred_raw, axis=1)

  # Calculate the confusion matrix.
  cm = confusion_matrix(y_test, test_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=class_names)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)

# Define the per-epoch callback.
cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)


############################################################################################
#################################### MODEL TRAINING ########################################
############################################################################################

# model training
classifier.fit(X_train, y_train, epochs=EPOCHS, 
    callbacks=[tensorboard_callback, cm_callback],
    validation_data=(X_test, y_test))

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

# execute tensorboard
os.system("tensorboard --logdir tensorboard/logs")