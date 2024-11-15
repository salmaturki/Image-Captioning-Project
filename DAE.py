import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

@tf.keras.utils.register_keras_serializable()
class CustomLossesAndMetrics:
    
    @staticmethod
    def ssim_loss(y_true, y_pred):
        # Compute the SSIM loss
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        return 1 - ssim

    @staticmethod
    def l1_loss(y_true, y_pred):
        # Compute the L1 loss
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    @staticmethod
    def combined_loss(y_true, y_pred):
        # Combine SSIM loss and L1 loss
        return CustomLossesAndMetrics.ssim_loss(y_true, y_pred) + CustomLossesAndMetrics.l1_loss(y_true, y_pred)
    
    @staticmethod
    def psnr_metric(y_true, y_pred):
        # PSNR metric
        return tf.image.psnr(y_true, y_pred, max_val=1.0)

    @staticmethod
    def ssim_metric(y_true, y_pred):
        # SSIM metric
        return tf.image.ssim(y_true, y_pred, max_val=1.0)

    @staticmethod
    def l1_metric(y_true, y_pred):
        # L1 metric
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    def get_config(self):
        return {}

@tf.keras.utils.register_keras_serializable()
class Encoder(layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        encoded = self.conv4(x)
        return encoded
    
    def get_config(self):
        # Returns the config of the Encoder
        config = super(Encoder, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Decoder Layer
@tf.keras.utils.register_keras_serializable()
class Decoder(layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.deconv1 = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.deconv2 = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.conv5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv6 = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')

    def call(self, inputs):
        x = self.deconv1(inputs)
        x = self.deconv2(x)
        x = self.conv5(x)
        decoded = self.conv6(x)
        return decoded
    
    def get_config(self):
        # Returns the config of the Decoder
        config = super(Decoder, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Autoencoder Model using Functional API
def build_autoencoder(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Pass through Encoder
    encoder = Encoder()
    encoded = encoder(inputs)
    
    # Pass through Decoder
    decoder = Decoder()
    decoded = decoder(encoded)
    
    # Build model
    autoencoder = Model(inputs, decoded)
    return autoencoder