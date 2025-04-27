import os
from PIL import Image

def find_corrupted_images(directory):
    corrupted_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img.verify()  # Verify integrity
                except (IOError, SyntaxError) as e:
                    corrupted_files.append(file_path)
    return corrupted_files

if __name__ == "__main__":
    dataset_path = "Mushrooms"  # Folder where your training images are
    broken_images = find_corrupted_images(dataset_path)
    
    print(f"Found {len(broken_images)} corrupted images.")
    for img in broken_images:
        print(img)
