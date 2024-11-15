#%%
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from DAE import Encoder, Decoder, CustomLossesAndMetrics
import matplotlib.pyplot as plt


# Load your pre-trained models
classification_model = load_model('./models/image_classification_model.keras')
denoising_model = load_model('./models/dae.keras', custom_objects={
    'Encoder': Encoder,
    'Decoder': Decoder,
    'combined_loss': CustomLossesAndMetrics.combined_loss,
    'psnr_metric': CustomLossesAndMetrics.psnr_metric,
    'ssim_metric': CustomLossesAndMetrics.ssim_metric,
    'l1_metric': CustomLossesAndMetrics.l1_metric
})
#%%
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess image to the correct input shape."""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img = tf.expand_dims(image.img_to_array(img)/255., axis=0)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def classify_image(img_array, model):
    """Classify image using the classification model."""
    prediction = model.predict(img_array)
    return np.argmax(prediction)

def denoise_image(img_array, model):
    """Denoise image using the denoising model."""
    denoised_img = model.predict(img_array)
    return denoised_img



def process_image_pipeline(image_path):
    # Step 1: Load and classify the image
    img_array = load_and_preprocess_image(image_path)
    initial_class = classify_image(img_array, classification_model)
    print(f'Initial class: {initial_class}')
    if initial_class == 1:  # 'Photo' class index is 1
        # Step 2: If classified as 'Photo', denoise and save it
        print('is photo')
        denoised_img = denoise_image(img_array, denoising_model)
        return denoised_img  # Save or use the denoised image

    else:
        # Step 3: If not 'Photo', denoise and reclassify
        print('not photo')
        denoised_img = denoise_image(img_array, denoising_model)
        reclassified_class = classify_image(denoised_img, classification_model)
        
        if reclassified_class == 1:  # If reclassified as 'Photo'
            # Optionally denoise again if necessary
            print('reclassified as photo')
            return denoised_img  # Save or use the denoised image
        else:
            print('reclassified as not photo')
            # Discard the image
            return None  # Image is not a 'Photo' after all

