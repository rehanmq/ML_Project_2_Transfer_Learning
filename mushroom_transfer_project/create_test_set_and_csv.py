import os
import shutil
import random
import csv
from PIL import Image

# Define paths
SOURCE_DIR = 'Mushrooms'               # your current dataset
TARGET_DIR = 'New_Test_File'           # new test folder
CSV_FILE = 'new_mushroom_test_file.csv'  # updated CSV name
NUM_IMAGES = 10                        # images per species
IMG_SIZE = (224, 224)                  # resize to match model input

# Create target directory if it doesn't exist
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

# Open CSV file to write
with open(CSV_FILE, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['filename', 'label'])  # header

    # Loop through each species folder
    for species in os.listdir(SOURCE_DIR):
        species_path = os.path.join(SOURCE_DIR, species)
        
        # Skip non-folder files (like .DS_Store)
        if not os.path.isdir(species_path):
            continue

        print(f'Processing species: {species}')
        
        # Create target species folder
        target_species_path = os.path.join(TARGET_DIR, species)
        os.makedirs(target_species_path, exist_ok=True)
        
        # Get list of images
        images = [img for img in os.listdir(species_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Randomly sample 10 images (or fewer if not enough)
        selected_images = random.sample(images, min(NUM_IMAGES, len(images)))
        
        for img_name in selected_images:
            img_path = os.path.join(species_path, img_name)
            try:
                # Open, resize, and save image
                img = Image.open(img_path)
                img = img.convert('RGB')  # ensure RGB
                img = img.resize(IMG_SIZE)
                
                # Save to new folder
                save_path = os.path.join(target_species_path, img_name)
                img.save(save_path)
                
                # Write entry to CSV (relative path)
                csv_writer.writerow([save_path, species])
            except Exception as e:
                print(f'Error processing {img_path}: {e}')

print('✅ New test file and new_mushroom_test_file.csv created successfully!')
