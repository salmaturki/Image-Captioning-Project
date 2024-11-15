import io
import os
# import mimetypes
import magic
import shutil
from PIL import Image, ImageOps
import tensorflow as tf

def verify_image(file_path):
    """
    Verifies the image by checking the MIME type, forcing load, and trying to re-save the image to clean
    any corrupt data.
    """
    # Detect the actual MIME type using python-magic
    mime = magic.Magic(mime=True)
    actual_mime_type = mime.from_file(file_path)
    
    # Supported MIME types and corresponding extensions
    mime_extension_map = {
        'image/jpeg': ['.jpg', '.jpeg'],
        'image/png': ['.png'],
        'image/gif': ['.gif'],
        'image/tiff': ['.tiff', '.jiff']
    }

    # Get the file extension
    extension = os.path.splitext(file_path)[1].lower()

    # Check if the detected MIME type matches the file extension
    if actual_mime_type not in mime_extension_map or extension not in mime_extension_map[actual_mime_type]:
        print(f"MIME type and extension mismatch for {file_path}: {actual_mime_type}")
        return False

    # Step 1: TensorFlow check
    try:
        img_bytes = tf.io.read_file(file_path)
        decoded_img = tf.io.decode_image(img_bytes)
    except tf.errors.InvalidArgumentError as e:
        print(f"TensorFlow failed on {file_path}... {e}")
        return False

    # Step 2: PIL verification and cleaning
    try:
        with Image.open(file_path) as img:
            # Handle EXIF orientation and color space issues
            img = ImageOps.exif_transpose(img)
            # skip if the image is not rgb
            if img.mode != 'RGB':
                print(f"Image {file_path} is not in RGB mode: {img.mode}")
                return False

            # Try to fully load the image
            img.load()

            # Step 3: Re-save the image to remove corrupt data
            output = io.BytesIO()
            img.save(output, format='JPEG')  # This re-saves the image to clean it
            output.seek(0)  # Reset buffer pointer
            Image.open(output).verify()  # Verify the cleaned image

    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return False

    return True

def process_images(directory, output_directory):
    """
    Processes all images in a directory (and subdirectories),
    verifies their validity, and copies valid images to the output directory,
    preserving the subdirectory structure.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Verify if the file is a valid image
            if verify_image(file_path):
                # Construct relative path and destination directory
                relative_path = os.path.relpath(root, directory)
                destination_dir = os.path.join(output_directory, relative_path)
                
                # Create the destination directory if it doesn't exist
                os.makedirs(destination_dir, exist_ok=True)
                
                # Copy the valid file to the new directory
                destination_file = os.path.join(destination_dir, file)
                shutil.copy2(file_path, destination_file)
                # print(f"Copied: {file_path} to {destination_file}")

def clean_dataset(input_dir, output_dir):
    """
    Main function to clean the dataset by verifying image files in the input directory
    and copying valid ones to the output directory.
    """
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    process_images(input_dir, output_dir)
    print(f"Dataset cleaning completed. Valid files copied to: {output_dir}")

# # Example usage
# input_directory = 'dataset'
# output_directory = 'dataset_clean4'

# clean_dataset(input_directory, output_directory)
