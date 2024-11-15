#%%
import tensorflow as tf

def load_model(filename):
    model = tf.keras.models.load_model(filename)
    return model


#%%
load_model('models/dae.keras')