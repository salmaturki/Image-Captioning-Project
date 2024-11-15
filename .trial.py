# %% [markdown]
# # Image Captioning Trials

# %%
%pip install -q -r requirements.txt

# %%
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential, Model
from onedrivedownloader import download
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import PIL
import datetime
import requests
from tensorboard.plugins.hparams import api as hp
import visualkeras
import zipfile
import json
from tqdm import tqdm
import collections
import nltk
from nltk.translate.bleu_score import corpus_bleu
import time

# Make sure to download NLTK data for BLEU if not already installed
nltk.download('punkt')

# %%
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

# %% [markdown]
# ## Loading the COCO dataset

# %%
annotations_url = "https://viacesifr-my.sharepoint.com/:u:/g/personal/salma_turki_viacesi_fr/EaPYZ3tgCY5CrjIskxHIn7sB2lSUjcIS9aTsMEucoIAyFw?e=omY7nF"
train2014_url = "https://viacesifr-my.sharepoint.com/personal/salma_turki_viacesi_fr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsalma%5Fturki%5Fviacesi%5Ffr%2FDocuments%2FCOCO%20Dataset%2Ftrain2014%2Ezip&parent=%2Fpersonal%2Fsalma%5Fturki%5Fviacesi%5Ffr%2FDocuments%2FCOCO%20Dataset&ga=1"

# URLs for the zip files
annotations_url = "https://viacesifr-my.sharepoint.com/:u:/g/personal/salma_turki_viacesi_fr/EaPYZ3tgCY5CrjIskxHIn7sB2lSUjcIS9aTsMEucoIAyFw?e=omY7nF"
train2014_url = "https://viacesifr-my.sharepoint.com/personal/salma_turki_viacesi_fr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsalma%5Fturki%5Fviacesi%5Ffr%2FDocuments%2FCOCO%20Dataset%2Ftrain2014%2Ezip&parent=%2Fpersonal%2Fsalma%5Fturki%5Fviacesi%5Ffr%2FDocuments%2FCOCO%20Dataset&ga=1"

# Local paths
annotation_folder = os.path.join(".", "annotations")
train2014_folder = os.path.join(".", "train2014")
annotation_zip = os.path.join(".", "annotations.zip")
train2014_zip = os.path.join(".", "train2014.zip")

# Check if the folders already exist
if not os.path.exists(annotation_folder):
    print("annotations folder not found. Downloading...")
    # Download the annotations.zip file
    response = requests.get(annotations_url, stream=True)
    with open(annotation_zip, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    del response

    # Unzip the annotations.zip
    print("Extracting annotations.zip...")
    with zipfile.ZipFile(annotation_zip, 'r') as zip_ref:
        zip_ref.extractall(annotation_folder)

    # Delete the annotations.zip file
    print("Deleting annotations.zip...")
    os.remove(annotation_zip)

if not os.path.exists(train2014_folder):
    print("train2014 folder not found. Downloading...")
    # Download the train2014.zip file
    response = requests.get(train2014_url, stream=True)
    with open(train2014_zip, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    del response

    # Unzip the train2014.zip
    print("Extracting train2014.zip...")
    with zipfile.ZipFile(train2014_zip, 'r') as zip_ref:
        zip_ref.extractall(train2014_folder)

    # Delete the train2014.zip file
    print("Deleting train2014.zip...")
    os.remove(train2014_zip)

print("Download and extraction completed.")

# %%
def get_feature_extraction_model(model_choice='InceptionV3'):
    """
    Returns the feature extraction model, preprocessing function, image size, and feature shapes based on the chosen model.
    """
    if model_choice == 'InceptionV3':
        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        img_size = (299, 299)  # Image size required for InceptionV3
        attention_features_shape = 64  # 8 * 8
        features_shape = 2048  # Depth of the feature map
    elif model_choice == 'ResNet50':
        image_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
        img_size = (224, 224)  # Image size required for ResNet50
        attention_features_shape = 49  # 7 * 7
        features_shape = 2048  # Depth of the feature map
    else:
        raise ValueError("model_choice must be either 'InceptionV3' or 'ResNet50'")

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    return image_features_extract_model, preprocess_input, img_size, features_shape, attention_features_shape


# %%
# model_choice = 'InceptionV3'  # Change this to 'ResNet50' to use ResNet
# image_features_extract_model, preprocess_input, img_size, features_shape, attention_features_shape = get_feature_extraction_model(model_choice)

# %%
model_choice = 'ResNet50'  # Change this to 'InceptionV3' to use ResNet
image_features_extract_model, preprocess_input, img_size, features_shape, attention_features_shape = get_feature_extraction_model(model_choice)

# %%
# Annotation file path
annotation_file = os.path.join(annotation_folder, "captions_train2014.json")

# Lecture du fichier d'annotation
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Group all annotations with the same identifier.
image_path_to_caption = collections.defaultdict(list)
for val in annotations['annotations']:
    # mark the beginning and end of each annotation
    caption = f"<start> {val['caption']} <end>"
    # An image's identifier is part of its access path.
    image_path = os.path.join(train2014_folder, 'COCO_train2014_' + '%012d.jpg' % (val['image_id']))
    # Add caption to image_path
    image_path_to_caption[image_path].append(caption)
    
# Take first images only
image_paths = list(image_path_to_caption.keys())
train_image_paths = image_paths[:5000]

# List of all annotations
train_captions = []
# List of all duplicated image file names (in number of annotations per image)
img_name_vector = []

for image_path in train_image_paths:
    caption_list = image_path_to_caption[image_path]
    # Add caption_list to train_captions
    train_captions.extend(caption_list)
    # Add duplicate image_path len(caption_list) times
    img_name_vector.extend([image_path] * len(caption_list))

print(f"Number of images: {len(train_image_paths)}")

# %%
# Function to load and preprocess image
def load_image(image_path):
    """
    Load and preprocess the image according to the model (InceptionV3 or ResNet50)
    The load_image function has as input the path of an image and as output a pair
    pair containing the processed image and its path.
    The load_image function performs the following processing:
        1. Loads the file corresponding to the path image_path
        2. Decodes the image into RGB.
        3. Resize image.
        4. Normalize image pixels between -1 and 1.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)  # Adjust image size based on model
    img = preprocess_input(img)  # Preprocess input according to selected model
    return img, image_path

# Image preprocessing for the dataset
encode_train = sorted(set(img_name_vector))

image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(8)

# Batch processing for feature extraction
for img, path in tqdm(image_dataset):
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())

# Display batch features shape
for img, path in image_dataset:
    batch_features = image_features_extract_model(img)
    print(f"Batch features shape: {batch_features.shape}")
    break

# %% [markdown]
# ## Pre-processing

# %% [markdown]
# ### Annotation pre-processing

# %%
# Find the maximum size
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Choose the 5000 most frequent words in the vocabulary
top_k = 5000
# The Tokenizer class enables text pre-processing for neural networks
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
# Builds a vocabulary based on the train_captions list
tokenizer.fit_on_texts(train_captions)

# Create the token used to fill annotations to equalize their length
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# Creation of vectors (list of integer tokens) from annotations (list of words)
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Fill each vector up to the maximum annotation length
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# Calculates the maximum length used to store attention weights
# It will later be used for display during evaluation.
max_length = calc_max_length(train_seqs)

# %% [markdown]
# ### Formation of a training and test set

# %%
img_to_cap_vector = collections.defaultdict(list)
# Creation of a dictionary associating image paths (.npy file) with annotations 
# # Images are duplicated because there are several annotations per image
print(len(img_name_vector), len(cap_vector))
for img, cap in zip(img_name_vector, cap_vector):
    img_to_cap_vector[img].append(cap)

"""
Creation of training and validation datasets 
using a random 80-20 split
""" 
# Take the keys (names of processed image files), *these will not be duplicated*.
img_keys = list(img_to_cap_vector.keys())
# Dividing clues into training and testing
slice_index = int(len(img_keys)*0.8)
img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

"""
Training and test games are in the form of
lists containing mappings:(pre-processed image ---> annotation token(word) )
"""

# Loop to build the training set
img_name_train = []
cap_train = []
for imgt in img_name_train_keys:
    capt_len = len(img_to_cap_vector[imgt])
    # Duplication of images by number of annotations per image
    img_name_train.extend([imgt] * capt_len)
    cap_train.extend(img_to_cap_vector[imgt])

# Loop to build the test set
img_name_val = []
cap_val = []
for imgv in img_name_val_keys:
    capv_len = len(img_to_cap_vector[imgv])
    # Duplication of images by number of annotations per image
    img_name_val.extend([imgv] * capv_len)
    cap_val.extend(img_to_cap_vector[imgv])

len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)

# %%
BATCH_SIZE = 32 # batch size
BUFFER_SIZE = 1000 # buffer size for data mixing
embedding_dim = 256
units = 512 # Hidden layer size in RNN
vocab_size = top_k + 1
num_steps = len(img_name_train) // BATCH_SIZE

# Function that loads numpy files from pre-processed images
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap

# Creation of a "Tensor "s dataset (used to represent large datasets)
# The dataset is created from "img_name_train" and "cap_train".
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load numpy files (possibly in parallel)
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Mixing data and dividing them into batches
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# %%
import tensorflow as tf
from abc import ABC, abstractmethod

# Base Encoder Class
class BaseEncoder(tf.keras.Model, ABC):
    @abstractmethod
    def call(self, x):
        pass

# Base Decoder Class
class BaseDecoder(tf.keras.Model, ABC):
    @abstractmethod
    def call(self, x, features, hidden):
        pass

# %% [markdown]
# ## CNN

# %%
class CNN_Encoder(BaseEncoder):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

# %% [markdown]
# ## Attention mechanisms

# %% [markdown]
# ### Bahdanau

# %%
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # W1 is the hidden layer for image features
        self.W1 = tf.keras.layers.Dense(units)
        # W2 is the hidden layer for the previous hidden layer
        self.W2 = tf.keras.layers.Dense(units)
        # V is the output layer that gives a non-normalized score for each image feature
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))

        # This gives you a non-normalized score for each image feature.
        score = self.V(attention_hidden_layer)

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

# %% [markdown]
# ## RNN

# %% [markdown]
# ### GRU

# %% [markdown]
# #### 1 layer

# %%
class RNN_Decoder_GRU(BaseDecoder):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder_GRU, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # self.gru = tf.keras.layers.GRU(self.units,
        #                                return_sequences=True,
        #                                return_state=True,
        #                                recurrent_initializer='glorot_uniform')
        self.gru = tf.keras.layers.RNN(tf.keras.layers.GRUCell(self.units), return_sequences=True, return_state=True)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        
        # Ensure x has a time dimension
        x = self.embedding(x)
        if len(x.shape) == 2:  # if shape is [batch_size, embedding_dim]
            x = tf.expand_dims(x, 1)  # expand dims to [batch_size, 1, embedding_dim]
        
        # Concatenate context vector with x
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Pass the concatenated tensor to the GRU
        output, state = self.gru(x)

        y = self.fc1(output)
        y = tf.reshape(y, (-1, y.shape[2]))  # Flatten before final Dense layer
        y = self.fc2(y)

        return y, state, attention_weights


    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# %% [markdown]
# #### 3 layers

# %%
class RNN_Decoder_GRU_3L(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder_GRU_3L, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # self.gru = tf.keras.layers.GRU(self.units, # Taille de la couche cachée du GRU
        #                                return_sequences=True, # retourne la séquence complète de sortie de chaque pas de temps
        #                                return_state=True, # retourne l'état caché de la dernière étape de temps
        #                                recurrent_initializer='glorot_uniform') # glorot_uniform est une initialisation des poids qui permet de mieux converger lors de l'entrainement en utilisant la fonction d'activation relu ce qui fait que les poids sont initialisés de manière à ce que la variance de la sortie soit égale à la variance de l'entrée
        # 3 couches de GRU
        self.gru1 = tf.keras.layers.RNN(tf.keras.layers.GRUCell(self.units), return_sequences=True, return_state=True)
        self.gru2 = tf.keras.layers.RNN(tf.keras.layers.GRUCell(self.units), return_sequences=True, return_state=True)
        self.gru3 = tf.keras.layers.RNN(tf.keras.layers.GRUCell(self.units), return_sequences=True, return_state=True)
        #Couche dense qui aura pour entrée la sortie du GRU
        self.fc1 = tf.keras.layers.Dense(self.units)
        # Dernière couche dense
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # L'attention est defini par un modèle a part
        context_vector, attention_weights = self.attention(features, hidden)
        # Passage du mot courant à la couche embedding
        x = self.embedding(x)
        # Concaténation
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1) #DONE tf.expand_dims permet de rajouter une dimension à un tenseur à une position donnée

        # Passage du vecteur concaténé à la gru
        output, state = self.gru1(x)
        output, state = self.gru2(output)
        output, state = self.gru3(output)
        
        # Couche dense
        y = self.fc1(output)

        y = tf.reshape(y, (-1, x.shape[2])) # Aplatir le tenseur pour le passer à la couche dense suivante (fc2)
        
        # Couche dense
        y = self.fc2(y)
        
        return y, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# %% [markdown]
# ### LSTM

# %%
class RNN_Decoder_LSTM(BaseDecoder):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder_LSTM, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, hidden_state, cell_state = self.lstm(x)
        y = self.fc1(output)
        y = tf.reshape(y, (-1, x.shape[2]))
        y = self.fc2(y)
        return y, hidden_state, attention_weights
    
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# %% [markdown]
# ## Training

# %%
# Create Model
def create_model(encoder_class, decoder_class, embedding_dim, units, vocab_size):
    encoder = encoder_class(embedding_dim)
    decoder = decoder_class(embedding_dim, units, vocab_size)
    return encoder, decoder

# Tokenization and BLEU evaluation functions
def tokenize_captions(captions, tokenizer):
    return [tokenizer.texts_to_sequences([cap])[0] for cap in captions]

def decode_predictions(preds, tokenizer):
    pred_captions = []
    for pred in preds:
        decoded_sentence = []
        for idx in pred:
            if idx == tokenizer.word_index['<end>']:
                break
            decoded_sentence.append(tokenizer.index_word[idx])
        pred_captions.append(decoded_sentence)
    return pred_captions

# %%
# Optimiseur ADAM
optimizer = tf.keras.optimizers.Adam() #DONE
# La fonction de perte
# SparseCategoricalCrossentropy est une fonction de perte qui est utilisée pour les problèmes de classification multi-classes. Elle est utilisée lorsque les étiquettes sont des entiers et non pas des vecteurs one-hot.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none') 

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# %%
# Validation step on separate validation dataset
loss_plot = []
@tf.function
def validation_step(img_tensor, target, encoder, decoder):
    loss = 0

    # Initialisation de l'état caché pour chaque batch
    hidden = decoder.reset_state(batch_size=target.shape[0])
    
    # Initialiser l'entrée du décodeur
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
    
    with tf.GradientTape() as tape: # Offre la possibilité de calculer le gradient du loss
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # Prédiction des i'èmes mot du batch avec le décodeur
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)

            # Le mot correct à l'étap i est donné en entrée à l'étape (i+1)
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1])) # Calcul de la perte moyenne par mot du batch courant

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss

# %%
# Optimizer and Checkpoint Management
checkpoint_path = "./checkpoints/train"

# %%
def train_model(encoder, decoder, ckpt_manager, start_epoch=0):
    # Modified training loop with BLEU score calculation
    EPOCHS = 5

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0
        
        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = validation_step(img_tensor, target, encoder, decoder)
            total_loss += t_loss

            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # sauvegarde de la perte
        loss_plot.append(total_loss / num_steps)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                            total_loss/num_steps))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # Affichage de la courbe d'entrainement
    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()

# %%
# Function to evaluate the image captioning model
def evaluate(image, encoder, decoder):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, attention_plot

# Function to display the BLEU score
def display_bleu_score(image, result):
    references = []

    # Display image
    image_show = PIL.Image.open(image)
    plt.imshow(image_show)
    plt.axis('off')
    plt.show()

    print("\n" + "*" * 60)
    print("Predicted Caption :")
    print(' '.join(result))

    # Find the corresponding references (captions) for this specific image
        # Find the corresponding references (captions) for this specific image
    for i, img_name in enumerate(img_name_val):
        if img_name == image:  # Compare with the image being processed
            ref = []
            for token in cap_val[i]:
                if token != 0:  # Ignore padding (0)
                    word = tokenizer.index_word.get(token, '<unk>')  # Handle missing tokens
                    ref.append(word)
            references.append(ref)
    print("\n" + "*" * 60)
    print("References :")
    for ref in references[:5]:
        print(' '.join(ref))

    print("...")

    # Convert references for BLEU score evaluation
    references_tokenized = [references]  # Multiple references for a single image

    # Since we're calculating BLEU for a single image, ensure predictions are used as a single list
    predictions = [result]  # Make the predicted caption a single hypothesis list

    # Calculate BLEU scores
    bleu_1 = corpus_bleu(references_tokenized, predictions, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(references_tokenized, predictions, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references_tokenized, predictions, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(references_tokenized, predictions, weights=(0.25, 0.25, 0.25, 0.25))

    # Print BLEU scores
    print("\n" + "*" * 60)
    print(f"BLEU Score :")
    print(f"unigram  = {bleu_1:.10f}")
    print(f"bigram   = {bleu_2:.10f}")
    print(f"trigram  = {bleu_3:.10f}")
    print(f"4-gram = {bleu_4:.10f}")
    print("*" * 60)

# Function for visualizing attention on the image
def plot_attention(image, result, attention_plot):
    temp_image = np.array(PIL.Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Model 1
# Workshop model (GRU) with BLEU and InceptionV3

# %%
# Instantiate encoder and decoder
encoder, decoder = create_model(CNN_Encoder, RNN_Decoder_LSTM, embedding_dim, units, vocab_size)

# Checkpoint setup
ckpt = tf.train.Checkpoint(encoder=encoder, 
                           decoder=decoder, 
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, 
                                          checkpoint_path, 
                                          max_to_keep=5)

# Resume training from last checkpoint if exists
start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)

# %%
train_model(encoder, decoder, ckpt_manager)

# %%
# Affichage de quelques annotations dans le jeu de test
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
print(image)
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])

result, attention_plot = evaluate(image, encoder, decoder)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)

# %%
display_bleu_score(image, result)
#%%
# play frequency sound when training is done (linux sound)
%system paplay /usr/share/sounds/freedesktop/stereo/complete.oga


# %%
# Save the Encoder model
encoder.save('models/captioning_models/encoder1_model.keras')

# Save the Decoder model
decoder.save('models/captioning_models/decoder1_model.keras')

# %% [markdown]
# ## Model 2
# Workshop model (GRU) with BLEU and ResNet50

# %%
# Instantiate encoder and decoder
encoder_resnet, decoder_resnet = create_model(CNN_Encoder, RNN_Decoder_GRU, embedding_dim, units, vocab_size)

# Checkpoint setup
ckpt = tf.train.Checkpoint(encoder=encoder_resnet, 
                           decoder=decoder_resnet, 
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, 
                                          checkpoint_path, 
                                          max_to_keep=5)

# Resume training from last checkpoint if exists
start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)

# %%
# train_model(encoder_resnet, decoder_resnet, ckpt_manager)

