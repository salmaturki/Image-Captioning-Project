from PIL import Image, UnidentifiedImageError
import os

def clean_image(file_path, output_path):
    try:
        # Open and verify the image to detect any potential corruption
        with Image.open(file_path) as img:
            img.verify()  # Verify the file integrity (detect corruption)
        
        # Reopen the image for further processing (necessary after calling verify)
        with Image.open(file_path) as img:
            img = img.convert('RGB')  # Fix unsupported color conversion issues
            # Re-save the image to clean it
            img.save(output_path, 'JPEG')
            # print(f"Cleaned and saved: {output_path}")
            return True
    except (UnidentifiedImageError, IOError, SyntaxError) as e:
        print(f"Corrupt or unsupported image: {file_path}, Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {file_path}, Error: {e}")
    return False

def clean_image_directory(directory, output_dir):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                
                # Preserve the subdirectory structure
                relative_path = os.path.relpath(root, directory)
                output_subdir = os.path.join(output_dir, relative_path)
                
                # Create output subdirectory if it doesn't exist
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                
                # Define the output file path
                output_path = os.path.join(output_subdir, file)
                
                if clean_image(file_path, output_path):
                    # print(f"Successfully processed: {file_path}")
                    pass
                else:
                    print(f"Skipping invalid image: {file_path}")
            else:
                print(f"Skipping unsupported file: {file}")

# Example usage
source_directory = "dataset"
output_directory = "dataset_clean_test"
clean_image_directory(source_directory, output_directory)
