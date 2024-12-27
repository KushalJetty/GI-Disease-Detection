import os
import shutil
import random

# Paths
source_database_path = ('./kvasir-dataset-v2')
test_folder_path = ('./test')


# Number of random images to select per classification
num_images_per_class = 20

# Ensure the test folder exists
os.makedirs(test_folder_path, exist_ok=True)

# Iterate through each classification subfolder
for class_folder in os.listdir(source_database_path):
    class_folder_path = os.path.join(source_database_path, class_folder)
    
    if os.path.isdir(class_folder_path):
        # Create corresponding subfolder in test folder
        test_class_folder_path = os.path.join(test_folder_path, class_folder)
        os.makedirs(test_class_folder_path, exist_ok=True)

        # Get all image files from the classification subfolder
        images = [f for f in os.listdir(class_folder_path) if os.path.isfile(os.path.join(class_folder_path, f))]
        
        # Randomly select the specified number of images
        selected_images = random.sample(images, min(num_images_per_class, len(images)))

        # Copy selected images to the test folder
        for image in selected_images:
            src_image_path = os.path.join(class_folder_path, image)
            dest_image_path = os.path.join(test_class_folder_path, image)
            shutil.copy(src_image_path, dest_image_path)

print("Test folder created successfully with randomly selected images.")
