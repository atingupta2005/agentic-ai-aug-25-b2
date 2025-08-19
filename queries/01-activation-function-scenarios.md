### Activation Function

* **Sigmoid**
    * **Description:** For binary classification (yes/no, 0/1) in the final layer.
    * **Example:** Predicting if an email is **spam or not spam** 📧. The final neuron outputs a value like 0.9, meaning there's a 90% probability that the email is spam.

* **Tanh**
    * **Description:** A zero-centered function, mostly used in older network architectures.
    * **Example:** In some older Recurrent Neural Networks (RNNs), Tanh was used in hidden layers to help **normalize values between -1 and 1**

* **ReLU**
    * **Description:** The most popular and default choice for hidden layers. It's simple and fast.
    * **Example:** In a network that recognizes images, the hidden layers use ReLU to learn features like edges, corners, and textures. It effectively "activates" a neuron only when a relevant feature is found.

* **Leaky ReLU**
    * **Description:** An improved version of ReLU that fixes the "dying neuron" problem.
    * **Example:** You're training a very deep network with ReLU, but the training gets stuck. You switch to Leaky ReLU to ensure all neurons stay slightly active, which can **help the network resume learning**.

* **PReLU**
    * **Description:** A smarter version of Leaky ReLU where the model learns the optimal slope.
    * **Example:** In a Generative Adversarial Network (GAN), where training stability is tricky, using PReLU can **give the model more flexibility**

* **ELU**
    * **Description:** An alternative to ReLU that can sometimes lead to faster learning.
    * **Example:** When building a model for a complex task like object detection, you might use ELU in the hidden layers. It can help the model converge on a good solution more quickly

* **Swish (SiLU)**
    * **Description:** A modern, high-performing alternative to ReLU, often used in very deep networks.
    * **Example:** Used in the hidden layers of modern computer vision models

* **Softmax**
    * **Description:** For multi-class classification (e.g., cat, dog, or horse) in the final layer.
    * **Example:** A model that identifies handwritten digits from 0 to 9. The final layer has 10 neurons, and Softmax outputs a probability for each digit. An output might look like `[0.05, 0.1, 0.7, 0.1, ...]` meaning the model is **70% sure the digit is a '2'**. 🔢