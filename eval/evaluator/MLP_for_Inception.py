import os

import tensorflow as tf
from tensorflow.keras.layers import (Flatten,
                                     Dense)
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, concatenate

if os.path.exists("/resources/"):
    RESOURCES_ROOT = "/resources/"
else:
    RESOURCES_ROOT = "../../resources/"


def simple_inception():
    input_img = Input(shape=(28, 28, 1))

    ### 1st layer
    layer_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input_img)
    layer_1 = Conv2D(10, (3, 3), padding='same', activation='relu')(layer_1)

    ### 3rd layer
    layer_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    layer_3 = Conv2D(10, (1, 1), padding='same', activation='relu')(layer_3)

    ### Concatenate
    mid_1 = concatenate([layer_1, layer_3], axis=3)

    flat_1 = Flatten()(mid_1)

    dense_1 = Dense(1200, activation='relu')(flat_1)
    dense_2 = Dense(600, activation='relu')(dense_1)
    dense_3 = Dense(150, activation='relu')(dense_2)

    output = Dense(10)(dense_3)

    model = Model([input_img], output)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def inception():
    input_img = Input(shape=(28, 28, 1))

    ### 1st layer
    layer_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input_img)
    layer_1 = Conv2D(10, (3, 3), padding='same', activation='relu')(layer_1)

    ### 2nd layer
    layer_2 = Conv2D(10, (1, 1), padding='same', activation='relu')(input_img)
    layer_2 = Conv2D(10, (5, 5), padding='same', activation='relu')(layer_2)

    ### 3rd layer
    layer_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    layer_3 = Conv2D(10, (1, 1), padding='same', activation='relu')(layer_3)

    ### Concatenate
    mid_1 = concatenate([layer_1, layer_2, layer_3], axis=3)

    flat_1 = Flatten()(mid_1)

    dense_1 = Dense(1200, activation='relu')(flat_1)
    dense_2 = Dense(600, activation='relu')(dense_1)
    dense_3 = Dense(150, activation='relu')(dense_2)

    output = Dense(10)(dense_3)

    model = Model([input_img], output)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


"""
Script used to create Classifier used in Fashion-MNIST inceoption score calculation.
"""


def create_classifier(dataset='fashionmnist'):
    print("CREATING CLASSIFIER FOR INCEPTION SCORE EVAL")
    if dataset == 'fashionmnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    else:
        raise NotImplementedError("dataset should be fashionmnist or mnist")

    train_images = train_images.astype('float32').reshape(-1, 28, 28, 1) / 255
    test_images = test_images.astype('float32').reshape(-1, 28, 28, 1) / 255

    model = simple_inception()

    checkpoint_path = RESOURCES_ROOT + f"/eval_models/own_inception_{dataset}_simple" + "/epochs/{epoch:02d}/cp.ckpt"
    print(f"Saving checkpoints to {checkpoint_path}")
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # Train the model with the new callback
    model.fit(train_images,
              train_labels,
              epochs=5,
              validation_data=(test_images, test_labels),
              callbacks=[cp_callback])  # Pass callback to training

    print(model.evaluate(test_images, test_labels))


if __name__ == '__main__':
    create_classifier('fashionmnist')
