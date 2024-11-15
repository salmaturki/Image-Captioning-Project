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

@tf.keras.utils.register_keras_serializable()
class CNN_Encoder(BaseEncoder):
    def __init__(self, embedding_dim=256, **kwargs):
        super(CNN_Encoder, self).__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
    
    def get_config(self):
        config = super(CNN_Encoder, self).get_config()
        # Add additional configurations here
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
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

    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({
            "W1": self.W1,
            "W2": self.W2,
            "V": self.V
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class RNN_Decoder_LSTM(BaseDecoder):
    def __init__(self, embedding_dim=256, units=512, vocab_size=5001, dropout_rate=0.3, **kwargs):
        super(RNN_Decoder_LSTM, self).__init__(**kwargs)
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden, training=False):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, hidden_state, cell_state = self.lstm(x)
        y = self.fc1(output)
        # Appliquer Dropout après la première couche fully-connected
        y = self.dropout(y, training=training)
        y = tf.reshape(y, (-1, x.shape[2]))
        y = self.fc2(y)
        return y, hidden_state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units)) 
    
    def get_config(self):
        config = super(RNN_Decoder_LSTM, self).get_config()
        config.update({
            "units": self.units,
            "embedding": self.embedding,
            "lstm": self.lstm,
            "dropout": self.dropout,
            "fc1": self.fc1,
            "fc2": self.fc2,
            "attention": self.attention
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)