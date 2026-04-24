import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# THIS IS THE PATH OF OUR DATASET
TRAIN_DIR = "archive/Training"
TEST_DIR = "archive/Testing"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# directories exist
if not os.path.exists(TRAIN_DIR):
    print("❌ Training folder not found:", TRAIN_DIR)
    exit()

if not os.path.exists(TEST_DIR):
    print("❌ Testing folder not found:", TEST_DIR)
    exit()

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Only rescale for testing
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Auto detect number of classes
num_classes = train_data.num_classes
class_names = list(train_data.class_indices.keys())

print("✅ Classes Found:", class_names)
print("✅ Total Classes:", num_classes)

# Transfer Learning Model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(train_data, epochs=EPOCHS, validation_data=test_data)

# Save model
model.save("brain_tumor_model.h5")
print("✅ Model Saved as brain_tumor_model.h5")

# Save class labels
with open("labels.txt", "w") as f:
    for label in class_names:
        f.write(label + "\n")

print("✅ Labels Saved as labels.txt")

# Plot Accuracy
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()