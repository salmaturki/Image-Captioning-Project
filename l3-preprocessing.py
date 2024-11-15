#%%
# import dependencies
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from images_check import clean_dataset
from onedrivedownloader import download
from py7zr import unpack_7zarchive
from sklearn.utils.class_weight import compute_class_weight


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        # Enable memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# %%
# Download dataset
filename = 'dataset.7z'
if not os.path.exists(filename):
    url = 'https://viacesifr-my.sharepoint.com/:u:/g/personal/arslane_ouldslimane_viacesi_fr/EYxF-OL6MXVIqY_Jps7fpGMBLUZzrsJajUZ3jHzgImkp_w?e=rvtSN6'

    download(
        url, 
        filename, 
        unzip=True,
        force_unzip=True,
        unzip_path='./datasets/',
        clean=True,
        )

# unzip in python
if not os.path.exists('./dataset/'):
    unpack_7zarchive(filename, './dataset/')

if not os.path.exists('./dataset_L2/'):
    os.rename('./dataset/Dataset-L2/', './dataset_L2/')

#%%
# load image classification dataset
ds_path = './dataset/'
ds_clean_path = './dataset_clean/'
# clean dataset
if not os.path.exists(ds_clean_path):
    clean_dataset(ds_path, ds_clean_path)
    
    
# %%
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 128
# load dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    ds_clean_path,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    ds_clean_path,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

#%%
train_ds = train_ds.map(lambda x, y: (x/255, y))
test_ds = test_ds.map(lambda x, y: (x/255, y))
# cache dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


#%%
def set_class_weights(dir_path):
    # Get class labels from dataset
    class_labels = sorted(os.listdir(dir_path))
    print("Classes:", class_labels)

    # Count the number of images per class
    count = {}
    for label in class_labels:
        count[label] = len(os.listdir(dir_path + label))
        print(f'{label}: {count[label]}')

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(class_labels),
        y=[label for label in class_labels for _ in range(count[label])]
    )

    # Assign class weights
    return {i: class_weights[i] for i in range(len(class_labels))}

class_weight_dict = set_class_weights(ds_clean_path)
print("Class Weights:", class_weight_dict)

#%%
data_augmentation = tf.keras.Sequential(
  [
    layers.RandomFlip(
                      "horizontal",
                      input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
                      ), 
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
  ]
)

def build_model():
    model = tf.keras.Sequential()
    # l1 regularization
    l1_lambda = 0.0
    # l2 regularization
    l2_lambda = 0.001
    # conditional data augmentation
    # if hp.Boolean('data_augmentation'):
    model.add(data_augmentation)
        
    # model.add(layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    
    # convolutional layers
    n_conv_layers = 2
    for i in range(1, n_conv_layers+1):
        model.add(layers.Conv2D(
            filters=2**(i+3),
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            strides=(2, 2),
            kernel_regularizer=regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)
        ))
        model.add(layers.MaxPooling2D(padding='same', pool_size=(2, 2)))
        
    
    # flatten layer
    model.add(layers.Flatten())
    
    # dense layers
    n_neurons = 448
    model.add(layers.Dense(n_neurons, activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)))
    
    # dropout layer
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(5, activation='softmax',
            kernel_regularizer=regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)))

    model.compile(
                    optimizer=tf.keras.optimizers.Adam(1e-3),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                  )
    
    return model

model = build_model()

#%%
# callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5
)

#%%
# train model
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50,
    callbacks=[early_stopping, lr_scheduler],
    class_weight=class_weight_dict
)
# %%
# plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()

#%%
model.evaluate(test_ds)

#%%
model.get_config()
# %%
model.save('models/image_classification_model.keras')
data_augmentation.save('models/data_augmentation.keras')

#%%
# load model
model = models.load_model('models/image_classification_model.keras')