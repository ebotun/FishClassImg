import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import math
import cv2
import os

# Img Dimensions
img_width, img_height = 150,150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

epochs = 10
batch_size = 16


def save_bottlebeck_features():
    # VGG16 Network Without the 3 last layers (include_top=Flase)
    model = applications.VGG16(include_top=False, weights='imagenet')

    train_datagen = ImageDataGenerator(rescale=1. / 255)

    generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))


    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(generator, predict_size_validation)

    np.save('bottleneck_features_validation.npy',bottleneck_features_validation)


def train_top_model():
    datagen_top_train = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
    generator_top = datagen_top_train.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use in predictions
    np.save('class_indices.npy', generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('bottleneck_features_train.npy')

    # get the class labels
    train_labels = generator_top.classes

    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    datagen_top_valid = ImageDataGenerator(rescale=1. / 255)

    generator_top = datagen_top_valid.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)


    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))



    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')


    checkpointer = ModelCheckpoint(filepath='etc/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5', verbose=1, save_best_only=False)




    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data,validation_labels),
                        shuffle=True,
                        verbose=0,
                        callbacks=[checkpointer])#,validation_split=0.2, callbacks=[early_stopping])
                       #

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("Accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("Loss: {}".format(eval_loss))


def predict():
    counter = 0
    correct = 0
    for i in os.listdir("Data/Data_test/"):
        counter +=1
        # load the class_indices saved in the earlier step
        class_dictionary = np.load('class_indices.npy').item()

        num_classes = len(class_dictionary)

        # add the path to your test image below
        image_path = "Data/Data_test/"+i
        image_name = i

        orig = cv2.imread(image_path)

        image = load_img(image_path, target_size=(150,150))
        image = img_to_array(image)

        # important! otherwise the predictions will be '0'
        image = image / 255

        image = np.expand_dims(image, axis=0)

        # VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')

        # get the bottleneck prediction from the pre-trained VGG16 model
        bottleneck_prediction = model.predict(image)

        # build top model
        model = Sequential()
        model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='sigmoid'))

        #Loading Weights, can use other path to use other weights created
        model.load_weights(top_model_weights_path)
        # use the bottleneck prediction on the top model to get the final
        # classification
        class_predicted = model.predict_classes(bottleneck_prediction)

        probabilities = model.predict_proba(bottleneck_prediction)

        inID = class_predicted[0]
        inv_map = {v: k for k, v in class_dictionary.items()}

        label = inv_map[inID]

        # get the prediction label
        print("ID: {} , Suggested Label: {}, Picture Label: {}".format(inID, label,image_name[:-4] ))

        if (int(inID)+1) == int(image_name[:-4]):
            correct += 1
    print("Accuracy of test_data: ",(correct/counter)*100)

#save_bottlebeck_features()
#train_top_model()

predict()

