import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image_dataset_from_directory

model_path = 'flower_model'
test_images_path = 'dataset/test'

model = tf.keras.models.load_model(model_path)

img_height = 180
img_width = 180
batch_size = 32

test_ds = image_dataset_from_directory(
    test_images_path,
    labels='inferred',
    label_mode='int',
    class_names=['cornflower', 'sunflower', 'wild poppie'],
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True)

class_names = test_ds.class_names

samples_per_class = 3
total_samples = samples_per_class * len(class_names)
sample_images = []
sample_labels = []

for images, labels in test_ds.unbatch().take(total_samples):
    sample_images.append(images)
    sample_labels.append(labels.numpy())

sample_images = np.stack(sample_images)
predictions = model.predict(sample_images)
predicted_classes = np.argmax(predictions, axis=1)


def plot_images(images, true_labels, predicted_classes, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(samples_per_class, len(class_names), i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(f"True: {class_names[true_labels[i]]}\nPredicted: {class_names[predicted_classes[i]]}")
        plt.axis("off")


plot_images(sample_images, sample_labels, predicted_classes, class_names)

plt.show()

all_true_labels = []
all_predictions = []
for images, labels in test_ds:
    all_true_labels.extend(labels.numpy())
    preds = model.predict(images)
    all_predictions.extend(np.argmax(preds, axis=1))

print("Classification Report:")
report = classification_report(all_true_labels, all_predictions, target_names=class_names)
print(report)
