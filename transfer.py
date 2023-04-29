# transfer.py
# Apply transfer learning to Xception net in order to predict which computer science professor is
# most likely to wear a given shirt.
#
# References:
# 1. https://arxiv.org/pdf/1610.02357.pdf
# 2. https://keras.io/api/applications/xception/
# 3. https://keras.io/guides/transfer_learning/

# Johnny Mixon and Raymond Riddell

import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from skimage.util import montage
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import argparse
import os
import pdb

ROOT = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(
    description="Using tranfer learning to correcly identify who would wear a particular shirt")
parser.add_argument('-r', '--ROOT', help="root directory location",
                    default=os.path.join(ROOT, './data'))
parser.add_argument('-v', '--verbose', action='store_true',
                    help="display updates in terminal")

DATAROOT = ROOT


def main(**kwargs):
    data_path = kwargs['ROOT']
    # Load dataset
    data, labels = load(data_path, show=kwargs['verbose'])
    labels, y = np.unique(labels, return_inverse=True)

    # Split data into random shuffled training (75%) and testing (25%) sets
    xtrain, xtest, ytrain, ytest = train_test_split(
        data, y, test_size=0.25, random_state=1)

    # Preprocess image data to match Xception requirements
    xtrain = tf.keras.applications.xception.preprocess_input(xtrain)
    xtest = tf.keras.applications.xception.preprocess_input(xtest)

    # Load the Xception model from Keras
    xception = tf.keras.applications.Xception(
        include_top=False,
        weights='imagenet',
        input_shape=xtrain.shape[1:]
    )

    # Change the end of the network to match new dataset (this is what will be learned!)

    avg = keras.layers.GlobalAveragePooling2D()(xception.output)
    output = keras.layers.Dense(len(labels), activation='softmax')(avg)

    #Make a new model with the old xception input and the new dense output

    model = keras.Model(xception.input, output)

    model.summary()
    input("Press Enter")
    # Freeze layers in xception
    for layer in xception.layers:
        layer.trainable = False

    # Compile the model
    optimizer = tf.keras.optimizers.experimental.SGD(
        learning_rate=0.1, momentum=0.9)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Create callbacks
    filename = os.path.join("transfer.h5")
    checkpoint = keras.callbacks.ModelCheckpoint(filename, save_best_only=True)
    earlystopping = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True)

    # Train the model using the fit method; use a batch size of 5, 100 epochs, the callbacks created above, and the test data for validation
    history = model.fit(xtrain, ytrain,
                        epochs=100,
                        batch_size=5,
                        shuffle=False,
                        use_multiprocessing=False,
                        validation_data=(xtest, ytest),
                        callbacks=(checkpoint, earlystopping),
                        verbose=kwargs['verbose'])

    # Evaluate the model
    model.evaluate(xtest, ytest)

    #get the predictions from the network 
    pred = model.predict(xtest)
    # sort each row by max since the network returns a confidence value for each class as a 2d array
    pred = np.argmax(pred, axis=1)

    # Compute the confusion matrix and show which shirts were misclassified
    print(confusion_matrix(pred, ytest))

    #create a mask of the misclassified shirts
    wrong_mask = pred != ytest
    #apply mask to xtest to get the incorrectly classified shirts
    wrong_img = xtest[wrong_mask]

    #just in case
    if len(wrong_img != 0):
        show_images((wrong_img))
    else:
        print("no wrong images!")


def load(directory=DATAROOT, show=False):
    '''Load data (and labels) from directory.'''
    files = os.listdir(directory)  # extract filenames
    n = len(files)  # number of files
    sz = [224, 224]  # required size of images for network

    # Read images from file and store in data array
    x = np.zeros([n] + sz + [3], dtype=int)
    for i in range(n):
        print(f'Loading data from file: {files[i]}')
        img = imread(os.path.join(directory, files[i]))
        if img.dtype == 'float32':
            img *= 255
        x[i] = np.array(tf.image.resize(img, sz))

    # Show montage of images (optional)
    if show:
        plt.figure(1)
        m = montage(x, fill=np.zeros(3), padding_width=1, multichannel=True)
        plt.imshow(m)
        plt.title('Sample Images')
        plt.xticks([])
        plt.yticks([])
        plt.show(block=False)

    # Extract labels from filenames
    y = np.array([file.split('.')[0][:-2] for file in files])
    return x, y


def show_images(x):
    '''shows incorrectly identified images'''
    plt.figure(2)
    m = montage(x, fill=np.zeros(3), padding_width=1, multichannel=True)
    plt.imshow((m / 2 + 0.5)) # image color data was ranging from -1 to 1 so did this to get it within bounds
    plt.title('Wrong Images')
    plt.xticks([])
    plt.yticks([])
    plt.show(block = True)


if __name__ == '__main__':
    main(**vars(parser.parse_args()))
