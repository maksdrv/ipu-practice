import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import matplotlib.pyplot as plt  
import keras
from keras.layers import Dense, Dropout
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
import h5py
train_labels_path = 'train-labels.idx1-ubyte'
train_images_path = 'train-images.idx3-ubyte'
test_labels_path = 't10k-labels.idx1-ubyte'
test_images_path = 't10k-images.idx3-ubyte'
EPOCHS = 5

@keras.saving.register_keras_serializable(package="my_package", name="f1_score")   
def f1_score(y_true, y_pred):
    precision = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_pred, 0, 1))) + K.epsilon())
    recall = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_true, 0, 1))) + K.epsilon())
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def plot_confusion(matrices):
    fig, axes = plt.subplots(2, 5, figsize = (25, 10))
    num = 0
    for i in range(2):
        for j in range(5):
            axes[i][j].set_title(f"{num}")
            sns.heatmap(matrices[num], 
            annot=True,
            ax = axes[i][j],
            xticklabels=[f'{num}',f'Not {num}'],
            yticklabels=[f'{num}',f'Not {num}'], fmt = 'g')
            axes[i][j].set_xlabel('Actual')
            axes[i][j].set_ylabel('Predicted')
            num += 1
    fig.savefig("confusion_matrix_classed.svg", format = "svg")

def confusion_matrices(y_test, pred):
    categorize = lambda item, num: 1 if item == num else 0
    cat_func = np.vectorize(categorize)
    matrices = []
    for num in range(10):
        cm = confusion_matrix(cat_func(y_test, num), cat_func(pred, num)) 
        matrices.append(cm)
    plot_confusion(matrices)

model = keras.Sequential([
        Dense(128, activation='relu', input_shape=(784, )),
        Dropout(0.2, input_shape=(784, )),
        Dense(64, activation='relu', input_shape=(784, )),
 #      Dense(128, activation='linear', input_shape=(784, )),
        Dense(10, activation='softmax')
    ])

def reading_labels_data(path):
    with open(path, 'rb') as f:
        train_labels_data = f.read()

    return np.frombuffer(train_labels_data, dtype = 'uint8', offset = 8)

def reading_images_data(path, length):
    train_images = []
    with open(path, 'rb') as f:
        f.read(4 * 4)
        for _ in range (length):
            image = f.read(28 * 28)
            image_np = np.frombuffer(image, dtype = 'uint8') / 255      # делим на 255, чтобы значения были от 0 до 1 (нормализация)
            train_images.append(image_np)

    return np.array(train_images)

def plot_image(pixels):
    plt.imshow(pixels.reshape((28, 28)), cmap=plt.cm.binary)
    plt.show()

def training(train_images, test_images, name):
  train_images = np.expand_dims(train_images.reshape(60000, 784), axis=2)
  test_images = np.expand_dims(test_images.reshape(10000, 784), axis=2)

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_score])
  his = model.fit(train_images, train_labels_classes, batch_size=1000, epochs=EPOCHS, validation_split=0.2) # обучение
  model.evaluate(test_images, test_labels_classes)
  model.save(name)


  #plt.plot(his.history['loss'], label = 'loss')
  #plt.title("График ошибок во время обучения")
  #plt.xlabel("Эпохи")
  #plt.ylabel("Ошибка")
  #plt.plot(his.history['val_loss'], label = 'val_loss')
  #plt.show()
 # plt.savefig()
  #print(his.history['accuracy'])
  table = np.zeros((3, EPOCHS))
  table[0], table[1], table[2] = his.history['loss'], his.history['accuracy'], his.history['f1_score']

  return table, his.history['accuracy'][EPOCHS-1]

train_labels = reading_labels_data(train_labels_path)
train_images = reading_images_data(train_images_path, 60000)
test_labels = reading_labels_data(test_labels_path)
test_images = reading_images_data(test_images_path, 10000)

train_labels_classes = keras.utils.to_categorical(train_labels, 10)
test_labels_classes = keras.utils.to_categorical(test_labels, 10)

tables =  np.zeros((6, 3, EPOCHS))
indexes = np.arange(784)

with h5py.File('datasets_and_model_accuracy.hdf5', 'a') as f:
  dset = f.create_dataset("noise_0/datasets/_train", data = train_images)
  dset = f.create_dataset("noise_0/datasets/_test", data = test_images)

t = 0
tables[t], _ = training(train_images, test_images, f"model_{t}.keras")
for s in [0.05, 0.1, 0.15, 0.2, 0.25]:
  t += 1
  for img in train_images.reshape(60000, 784):
    random.shuffle(indexes)
    for i in indexes[:int(s*784)]:
      img[i] = 1

  for img in test_images.reshape(10000, 784):
    random.shuffle(indexes)
    for i in indexes[:int(s*784)]:
      img[i] = 1

  with h5py.File('datasets_and_model_accuracy.hdf5', 'a') as f:
     f.create_dataset(f"noise_{int(s*100)}/datasets/_train", data = train_images)
     f.create_dataset(f"noise_{int(s*100)}/datasets/_test", data = test_images)
  print("!!!")
  tables[t], _ = training(train_images, test_images, f"model_{t}.keras")


with h5py.File('datasets_and_model_accuracy.hdf5', 'a') as f:
  f.create_dataset("noise_0/loss_and_accuracy", data=tables[0])
  f.create_dataset("noise_5/loss_and_accuracy", data=tables[1])
  f.create_dataset("noise_10/loss_and_accuracy", data=tables[2])
  f.create_dataset("noise_15/loss_and_accuracy", data=tables[3])
  f.create_dataset("noise_20/loss_and_accuracy", data=tables[4])
  f.create_dataset("noise_25/loss_and_accuracy", data=tables[5])

print(tables)

matrix = np.zeros((6,6))
#tables =  np.zeros((6, 6, 2, EPOCHS))
i = 0

for n in [0, 0.05, 0.1, 0.15, 0.2, 0.25] :
  j = 0
  with h5py.File("datasets_and_model_accuracy.hdf5", 'r') as f:
    train_im = f[f"noise_{int(n*100)}/datasets/_train"]
    traim_images = train_im[:]
  for m in [0, 0.05, 0.1, 0.15, 0.2, 0.25]:
    with h5py.File("datasets_and_model_accuracy.hdf5", 'r') as f:
      test_im = f[f"noise_{int(m*100)}/datasets/_test"]
      test_images = test_im[:]
    _, matrix[i][j] = training(train_images, test_images, f"model_{t}.keras")
    j += 1
  i += 1

print(matrix)

with h5py.File("datasets_and_model_accuracy.hdf5", 'a') as f:
  f.create_dataset("table_models_accuracy", data = matrix)

with h5py.File("table_accuracy.hdf5", 'a') as f:
  f.create_dataset("table", data = matrix)





