import numpy as np
import tensorflow as tf
import random as rn
import os

RANDOM_SEED = 12345
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.utils import *
from Utils import  *
import math
import pickle
import matplotlib.pyplot as plt

COMBINE_MODEL_BATCH_SIZE = 300

class GenresInputGenerator(Sequence):
    def __init__(self, path, batch_size = COMBINE_MODEL_BATCH_SIZE):
        self.path = path
        self.batch_size = batch_size
        self.class_dict = { 'blues' : 0, 'classical' : 1, 'country' : 2,  'disco' : 3, 'hiphop' : 4,
                            'jazz' : 5, 'metal' : 6,'pop' : 7, 'reggae' : 8, 'rock' : 9}
        self.file_list = []
        self.current_file_index = 0
        self.create_file_list()

    def __getitem__(self, batch_index):
        batch_x = np.empty((self.batch_size, 20))
        batch_y = np.empty(self.batch_size)

        for i in range(self.batch_size):
            file_index = self.batch_size * batch_index + i
            file_name, class_name = self.file_list[file_index]
            class_index = self.class_dict[class_name]
            x = np.load(file_name)
            batch_x[i,] = x
            batch_y[i] = class_index

        return batch_x, to_categorical(batch_y, num_classes=10)

    def __len__(self):
        return math.floor(len(self.file_list) / self.batch_size)

    def create_file_list(self):
        for dir in os.listdir(self.path):
            class_dir = os.path.join(self.path, dir)
            for filename in os.listdir(class_dir):
                if filename.endswith('.npy'):
                    class_name = os.path.basename(class_dir)
                    self.file_list.append((class_dir + "/" +filename, class_name))

    def on_epoch_end(self):
        np.random.shuffle(self.file_list)

def create_model():
    model = Sequential()
    model.add(Dense(10, input_shape=(20,)))
    model.add(Activation('softmax'))
    opt = optimizers.Adam(lr=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train(model):
    train_generator = GenresInputGenerator(COMBINED_INPUT_DIR + 'train/')
    test_generator = GenresInputGenerator(COMBINED_INPUT_DIR + 'test/')

    checkpoint = ModelCheckpoint('Models/combineModel.h5', save_weights_only=False, save_best_only=True, period=1,
                                 verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=100, verbose=1, mode='auto')

    history = model.fit_generator(
        generator=train_generator,
        validation_data=test_generator,
        epochs=1000,
        callbacks=[checkpoint, reduce_lr, early_stopping]
    )

    with open('Logs/combineModelHistory', 'wb') as file:
        pickle.dump(history.history, file)

def test(model):
    train_generator = GenresInputGenerator(COMBINED_INPUT_DIR + 'train/')
    test_generator = GenresInputGenerator(COMBINED_INPUT_DIR + 'test/')
    print('Testing train set...')
    train_score = model.evaluate_generator(generator=train_generator)
    print('Loss: {0} - Acc: {1}'.format(train_score[0], train_score[1]))
    print('Testing test set...')
    test_score = model.evaluate_generator(generator=test_generator)
    print('Loss: {0} - Acc: {1}'.format(test_score[0], test_score[1]))

def draw_history():
    history = pickle.load(open('Logs/combineModelHistory', 'rb'))
    plt.plot(history['acc'], 'b--', history['val_acc'], 'r--', history['loss'], 'b', history['val_loss'], 'r')
    plt.legend(('Train accuracy', 'Test accuracy', 'Train cost function value', 'Test cost function value'))
    plt.xlabel('Epochs')
    plt.show()

def train_combine_model():
    model = create_model()
    model.summary()
    train(model)
    test(model)
    draw_history()