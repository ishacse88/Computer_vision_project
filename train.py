import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, RandomFlip, RandomRotation
from tensorflow.keras.models import Model
import os

# 1. Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_DIR = 'model'

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Load Dataset
print("Loading datasets...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Detected classes: {class_names}")

# 3. Data Augmentation & Preprocessing Pipeline
data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
])

# MobileNetV2 expects pixel values between -1 and 1
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# 4. Build the Transfer Learning Model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)

# Load pre-trained MobileNetV2 (excluding the final classification layer)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Freeze base model for initial training

x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)

# Output layer (Softmax for probabilities)
outputs = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs, outputs)

# 5. Compile and Train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Starting training...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# 6. Save the model for the Streamlit app
model.save(f'{MODEL_DIR}/fruit_quality_model.keras')
print("Model saved successfully!")