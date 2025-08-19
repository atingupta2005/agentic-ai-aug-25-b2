# A Guide to Pre-designed Neural Network Architectures

Instead of building a network from scratch, you can use a professionally designed, ready-made architecture. This saves time and leverages years of research. There are two primary ways to use these models.

-----

## Two Ways to Use Pre-designed Architectures

Think of an architecture as a **blueprint** 🏗️ and the trained weights as the **learned knowledge** 🧠. You can choose to use the blueprint with or without the pre-existing knowledge.

### 1\. Transfer Learning (Blueprint + Knowledge)

You use a model that has already been trained on a huge dataset (like ImageNet). The knowledge learned from that dataset is transferred to your new task.

  * **When to use it:** When your task is **similar** to the original training data (e.g., classifying photos of everyday objects) and you have a **limited amount of your own data**.

### 2\. Training from Scratch (Blueprint Only)

You use only the architecture (the blueprint) of a famous model but initialize it with random weights. You then train it entirely on your own data.

  * **When to use it:** When your dataset is **very unique** (e.g., medical X-rays, financial charts, satellite imagery) or when you have a **very large dataset** of your own.

| Approach | Best For | Key Idea |
| :--- | :--- | :--- |
| **Transfer Learning** | Similar tasks, smaller datasets | Use the pre-learned knowledge as a powerful head start. |
| **Training from Scratch**| Unique tasks, larger datasets | Use the proven blueprint but learn everything from your own data. |

-----

## A Catalog of Popular Architectures (The Blueprints)

Here is a list of popular, non-transformer architectures, primarily from computer vision, that you can import and use.

### The VGG Family

A classic family known for its simple, uniform architecture of repeating blocks.

  * **VGG16**, **VGG19**

-----

### The ResNet Family

The workhorse of computer vision. It introduced "skip connections" to allow for training extremely deep and powerful models. A fantastic default choice.

  * **ResNet50**, **ResNet101**, **ResNet152**
  * **ResNet50V2**, **ResNet101V2**, **ResNet152V2**

-----

### The Mobile & Efficient Family

Designed to be lightweight, fast, and effective on devices with limited power, like smartphones 📱.

  * **MobileNet**, **MobileNetV2**, **MobileNetV3Small**, **MobileNetV3Large**
  * **EfficientNetB0**, **EfficientNetB1** up to **B7** (modern models that offer state-of-the-art accuracy for their size)

-----

### The Inception & Xception Family

Focuses on computational efficiency by performing convolutions at different scales in parallel.

  * **InceptionV3**
  * **InceptionResNetV2** (combines Inception with skip connections)
  * **Xception** (an evolution of the Inception idea)

-----

### The DenseNet Family

Uses "dense connections" where each layer connects to every subsequent layer, which improves feature reuse.

  * **DenseNet121**, **DenseNet169**, **DenseNet201**

-----

## Practical Guide: How to Import and Use Them

### For Computer Vision Models (Keras / PyTorch)

**1. Using Transfer Learning (with pre-trained weights)**
You load the model and specify the pre-trained weights (e.g., `'imagenet'`).

```python
from tensorflow.keras.applications import ResNet50

# Load ResNet50 with weights pre-trained on the ImageNet dataset
model = ResNet50(weights='imagenet')

# The model is now ready to make predictions
```

**2. Training from Scratch (architecture only)**
The key is to set **`weights=None`**. This gives you the blueprint with random weights, ready to be trained.

```python
from tensorflow.keras.applications import EfficientNetB0

# Load the EfficientNetB0 ARCHITECTURE ONLY.
model = EfficientNetB0(weights=None, classes=10, input_shape=(224, 224, 3))

# Now, you can compile and train this proven architecture on your own data
# model.compile(optimizer='adam', loss='categorical_crossentropy')
# model.fit(your_dataset, ...)
```

-----

## Which Architecture Should I Choose?

Here's a quick guide to help you decide:

  * **For a strong, reliable default:** Start with **ResNet50**. It's a fantastic and well-understood baseline.
  * **For mobile or speed-sensitive applications:** Use **MobileNetV2** or **MobileNetV3**. They are designed to be fast and lightweight.
  * **For the best possible accuracy:** Try the **EfficientNet** family (start with B0 and go up). They often provide the best performance for a given size.