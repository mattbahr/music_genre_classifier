import tensorflow as tf
import numpy as np
import pathlib
import os

training_dir = pathlib.Path('/data/training_set')
test_dir = pathlib.Path('/data/test_set')

batch_size = 32
img_height = 288
img_width = 432
img_channels = 3
epochs = 5

pretrained = False
training = False

if not pretrained:
    train_ds = tf.keras.utils.image_dataset_from_directory(
      training_dir,
      validation_split=0.05,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
      training_dir,
      validation_split=0.05,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
else:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        training_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=1)

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

if not pretrained:
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

if not pretrained:
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 10

if pretrained:
    model = tf.keras.models.load_model('/src/model')
    training = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(train_ds, epochs=epochs, verbose=2)
    test_loss, test_acc = model.evaluate(test_ds)

    if test_acc > 0.7085:
        model.save('/src/model')
elif not training and os.path.exists('/src/model'):
    model = tf.keras.models.load_model('/src/model')
    training = False
else:
    training = True
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 7, activation='relu', input_shape=[img_height, img_width, img_channels]),
      tf.keras.layers.MaxPool2D(),
      tf.keras.layers.Conv2D(64, 5, activation='relu'),
      tf.keras.layers.MaxPool2D(),
      tf.keras.layers.Conv2D(128, 3, activation='relu'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.summary()

    initial_learning_rate = 0.001
    decay_steps = 100000
    decay_rate=0.96

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)

    model.compile(
      optimizer=tf.keras.optimizers.Nadam(),
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])

    model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs,
      verbose=2)

    test_loss, test_acc = model.evaluate(test_ds)

    if test_acc > 0.6533:
        print('Saving model...')
        model.save('/src/model')

if not training:
    model.evaluate(test_ds)

x_test = []
y_test = []

for x, y in test_ds:
    x_test.append(x)
    y_test.append(y)

for i in range(len(x_test)):
    y_pred = model.predict(x_test[i])
    print('Label', y_test[i], 'Prediction:', np.argmax(y_pred))
