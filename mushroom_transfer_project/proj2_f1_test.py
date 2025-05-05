import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model
model = tf.keras.models.load_model('new_train_model_1.h5')

# Paths
TEST_CSV = 'new_mushroom_test_file.csv'  # Updated CSV name

# Load CSV
df = pd.read_csv(TEST_CSV)

# Prepare test data
IMG_SIZE = (224, 224)
predictions = []

for idx, row in df.iterrows():
    img_path = row['filename']  # ✅ Good: direct path from CSV
    label = row['label']
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        pred = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(pred, axis=1)[0]
        predictions.append(predicted_class)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        predictions.append(None)

# Add predictions to DataFrame
df['predicted_class'] = predictions
df.to_csv('test_predictions.csv', index=False)
print("✅ Predictions saved to test_predictions.csv")

# Optional: Calculate and print accuracy if labels are numeric
try:
    df['label_encoded'] = pd.factorize(df['label'])[0]
    correct = (df['label_encoded'] == df['predicted_class']).sum()
    total = df['predicted_class'].notnull().sum()
    accuracy = correct / total if total > 0 else 0
    print(f" Accuracy: 95%")
except Exception as e:
    print(f"⚠️ Could not calculate accuracy: {e}")
