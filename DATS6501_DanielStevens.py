####Import Packages####
from glob import glob
import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import datetime
import matplotlib.pyplot as plt
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Conv2D, Flatten, Reshape, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.models import load_model

# %% --------------------------------------- Set-Up Variables --------------------------------------------------------------------
# Configures environment variables and elements of reproducibility
LR = 1e-4
N_NEURONS = (1200, 900, 900)
N_EPOCHS = 20
BATCH_SIZE = 200
DROPOUT = 0.2
SAMPLE_COUNT = 7929
WIDTH = 100
HEIGHT = 100
COLOR = True
TEST_SIZE = .15
ACT_FUNC_1 = 'relu'
ACT_FUNC_2 = 'relu'
LOSS_FUNC = 'binary_crossentropy'
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)
runstat = {}
runstat['DTG_START'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
runstat['LR'] = LR
runstat['N_NEURONS'] = str(N_NEURONS)
runstat['N_EPOCHS'] = N_EPOCHS
runstat['BATCH_SIZE'] = BATCH_SIZE
runstat['DROPOUT'] = DROPOUT
runstat['SAMPLE_COUNT'] = SAMPLE_COUNT
runstat['HEIGHT'] = HEIGHT
runstat['WIDTH'] = WIDTH
runstat['COLOR'] = COLOR
runstat['TEST_SIZE'] = TEST_SIZE
runstat['ACT_FUNC_1'] = ACT_FUNC_1
runstat['ACT_FUNC_2'] = ACT_FUNC_2
runstat['LOSS_FUNC'] = LOSS_FUNC

#Directory Location
print(os.getcwd())

# %% ----------------------------------- Read Labels  --------------------------------------------------------------
bees=pd.read_csv(os.getcwd()+'/bee_data_new.csv',
                index_col=False,
                parse_dates={'datetime':[1,2]},
                dtype={ 'health':'category'})
bees = bees.sample(300)


# %% ----------------------------------- Image Manipulation  --------------------------------------------------------------
#Directory Location
print(os.getcwd())
img_folder=os.getcwd()+ '/bee_imgs/bee_imgs/'


# Keep only images with bees
img_exists = bees['file'].apply(lambda f: os.path.exists(img_folder + f))
bees = bees[img_exists]
print(bees.head())


# Generate more data by rotating and flipping images
image_files = glob('bee_imgs/bee_imgs/0*.png')
print(str(len(image_files)) + ' training images found..')
images = []
labels = []
rotated90 = []
flipped = []
i = 0
for image in image_files[:SAMPLE_COUNT]:
    i += 1
    #label1 = float(image.split('_')[1])
    label1 = bees['health'].cat.categories
    # Load and resize image
    im1 = cv2.imread(image)
    ri0 = cv2.resize(im1, dsize=(WIDTH, WIDTH), interpolation=cv2.INTER_CUBIC)
    # Rotate and flip images
    rotate90 = cv2.warpAffine(ri0, cv2.getRotationMatrix2D((WIDTH/2, HEIGHT/2), 90, 1), (WIDTH, HEIGHT))
    rotate180 = cv2.warpAffine(ri0, cv2.getRotationMatrix2D((WIDTH/2, HEIGHT/2), 180, 1), (WIDTH, HEIGHT))
    rotate270 = cv2.warpAffine(ri0, cv2.getRotationMatrix2D((WIDTH/2, HEIGHT/2), 270, 1), (WIDTH, HEIGHT))
    hi0 = cv2.flip( ri0, 0 )
    vi0 = cv2.flip( ri0, 1 )
    hvi0 = cv2.flip( ri0, -1 )
    images += [ri0, rotate90, rotate180, rotate270, hi0, vi0, hvi0]
    rotated90 += [rotate90]
    flipped += [vi0]
    labels += [label1] * 7


# %% ----------------------------------- Train Test Split --------------------------------------------------
####CNN Train test split####
SAMPLE_COUNT = len(images)
imgarray= np.array(images)
print(imgarray.shape)
print(str(SAMPLE_COUNT) + ' total images')
labels_array = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(imgarray, labels_array, test_size=TEST_SIZE, shuffle=False)
print(x_train.shape)
print(x_test.shape)

####MLP Train test split####
images_array1 = np.concatenate(images, axis=0).reshape(SAMPLE_COUNT,WIDTH*HEIGHT*3)
labels_array1 = np.array(labels)
x_train1, x_test1, y_train1, y_test1 = train_test_split(images_array1, labels_array1, test_size=TEST_SIZE, shuffle=False)

# %% -------------------------------------- Training Prep ------------------------------------------------
#####Train the Covolutional Neural Network#####
model_1 = Sequential()
model_1.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(100, 100, 3)))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(BatchNormalization())
model_1.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(BatchNormalization())
model_1.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(BatchNormalization())
model_1.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(BatchNormalization())
model_1.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(BatchNormalization())
model_1.add(Dropout(0.2))
model_1.add(Flatten())
model_1.add(Dense(2, activation='sigmoid'))
model_1.compile(optimizer=Adam(lr=LR), loss='binary_crossentropy', metrics=[metrics.categorical_accuracy, metrics.mae, metrics.mse])

#####Train the MultiLayer Perceptron#####
model_2 = Sequential([
    Dense(N_NEURONS[0], kernel_initializer=weight_init),
    Activation(ACT_FUNC_1),
    Dropout(DROPOUT),
    BatchNormalization()
])
# Loop over the hidden dimensions to add more layers
for n_neurons in N_NEURONS[1:]:
    model_2.add(Dense(n_neurons, activation=ACT_FUNC_2, kernel_initializer=weight_init))
    model_2.add(Dropout(DROPOUT, seed=SEED))
    model_2.add(BatchNormalization())
# Add final output layer with sigmoid
model_2.add(Dense(2, activation="sigmoid", kernel_initializer=weight_init))
model_2.compile(optimizer=Adam(lr=LR), loss=LOSS_FUNC, metrics=[metrics.mae, metrics.mse])


# %% -------------------------------------- Training Epochs ----------------------------------------------------------
# Trains the model, while printing validation metrics at each epoch
model_1.fit(x_train, y_train,batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test), shuffle=True,
          callbacks=[ModelCheckpoint("cnn_bee.hdf5", monitor="val_loss", save_best_only=True)])

model_2.fit(x_train1, y_train1, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test1, y_test1), shuffle=True,
          callbacks=[ModelCheckpoint("mlp_bee.hdf5", monitor="val_loss", save_best_only=True)])

#plot loss for both models
plt.plot(model_1.history.history['loss'])
plt.plot(model_1.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('cnn_loss.png')
plt.plot(model_2.history.history['loss'])
plt.plot(model_2.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('mlp_loss.png')

#%% ------------------------------------------ Final Test & Results -------------------------------------------------------------
# Load best version of stronger model and calculate final score
model_1 = load_model('cnn_bee.hdf5')
runstat['DTG_STOP'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
runstat['Loss'] = model_1.evaluate(x_test, y_test)[0]
print("Final loss on validations set:", runstat['Loss'])
print(model_1.summary())


