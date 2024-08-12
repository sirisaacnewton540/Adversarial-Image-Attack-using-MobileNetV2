# **Adversarial Image Attack using MobileNetV2**

## **Overview**

This project demonstrates the implementation of adversarial attacks on image classification models, specifically using the Fast Gradient Sign Method (FGSM) on a pre-trained MobileNetV2 model. Adversarial attacks are small, often imperceptible perturbations added to input images that can deceive a neural network, causing it to misclassify the image. This project includes a user-friendly GUI to upload images and visualize the effects of adversarial perturbations.

## **Key Features**

- **Adversarial Attack Implementation:** Utilizes FGSM to create adversarial examples that fool the MobileNetV2 model.
- **Image Classification:** The project displays the top prediction for both original and adversarial images.
- **GUI Integration:** A simple tkinter-based GUI allows users to upload images, view the perturbations, and see how the adversarial attack impacts the model's predictions.
- **Educational Focus:** Designed for those interested in understanding adversarial attacks, image classification vulnerabilities, and neural network security.

## Results
![Screenshot (652)](https://github.com/user-attachments/assets/cdbe737e-f3f2-44f6-af02-4ed723ce386e)
![Screenshot (653)](https://github.com/user-attachments/assets/423ae020-6382-4056-ac2d-d58e453b30ea)
![Screenshot (651)](https://github.com/user-attachments/assets/b186e5bf-6871-437c-b6a0-28d3a8645cd0)

This project visually demonstrates the vulnerability of deep learning models to adversarial attacks. After running the application and uploading an image, you will observe the following results:

### **1. Original Image**
- **Description:** This is the input image as it was originally passed to the MobileNetV2 model. The model processes this image and provides a classification prediction based on what it has learned during its training on the ImageNet dataset.
- **Predicted Class:** The class label displayed on the top left of the original image represents the highest confidence prediction made by the model. This is the model’s best guess at what the image represents, assuming the image is unaltered.

### **2. Perturbations**
- **Description:** This image represents the calculated perturbations that are added to the original image to create the adversarial image. These perturbations are deliberately crafted to be minimal yet effective in deceiving the neural network. While these changes are often imperceptible to the human eye, they are enough to manipulate the model's prediction.
- **Purpose:** The perturbations demonstrate how small changes in the input data can lead to a significant shift in the model's output. The perturbations are calculated by the Fast Gradient Sign Method (FGSM), which alters the image in a way that maximizes the model's prediction error.

### **3. Adversarial Image**
- **Description:** This is the final output after adding the perturbations to the original image. Visually, the adversarial image may appear almost identical to the original image; however, the subtle changes introduced by the perturbations are sufficient to fool the neural network.
- **Predicted Class:** The class label shown on the adversarial image may differ from the original image’s class, showcasing the success of the adversarial attack. Even though the image looks the same to a human, the model may now classify it incorrectly.

### **Understanding the Impact:**
- **Model Vulnerability:** The fact that the model can be misled by such minimal changes highlights a critical weakness in neural networks. This has broad implications, especially in security-sensitive applications like autonomous driving or facial recognition, where misclassifications can have serious consequences.
- **Adversarial Robustness:** These results underscore the importance of developing and incorporating adversarial training and other defense mechanisms into neural network models to enhance their robustness against such attacks.

### **Key Takeaways:**
- **Visual Similarity vs. Model Interpretation:** The project illustrates how neural networks "see" images differently from humans. Even when two images look identical to us, the model may interpret them very differently, leading to incorrect predictions.
- **The Importance of Security in AI:** Understanding and mitigating adversarial attacks is crucial in the deployment of AI models in real-world applications. This project serves as a practical demonstration of why robust AI models are essential for the safe and reliable use of AI technologies.


## **Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Understanding Adversarial Attacks](#understanding-adversarial-attacks)
- [Fast Gradient Sign Method (FGSM)](#fast-gradient-sign-method-fgsm)
- [Model Used: MobileNetV2](#model-used-mobilenetv2)
- [How the Code Works](#how-the-code-works)
- [Potential Application](#potential-application)
- [Further Reading](#further-reading)
- [Contributing](#contributing)

## **Installation**

**Install Dependencies:**
   Ensure you have Python 3.x installed. Install the necessary Python packages:
   ```bash
   pip install tensorflow pillow matplotlib
   ```

## **Usage**

1. **Upload an Image:**
   - Click on the "Load Image" button in the GUI.
   - Select an image from your local machine. The image will be displayed alongside its adversarial counterpart and the perturbations.

2. **Visualize Results:**
   - The GUI will display the original image, the perturbations, and the adversarial image side by side.
   - The predicted class for the original image is shown on the top left of the original image.

## **Understanding Adversarial Attacks**

Adversarial attacks exploit the vulnerabilities of neural networks by introducing minimal perturbations to input data, causing the model to make incorrect predictions. These perturbations are often imperceptible to the human eye, yet they can lead to drastically different outputs from the model.

### **Why Study Adversarial Attacks?**
- **Security Implications:** Understanding these attacks is crucial in fields like autonomous driving, facial recognition, and healthcare, where misclassifications can have severe consequences.
- **Model Robustness:** By studying adversarial attacks, researchers aim to develop more robust models that can withstand such manipulations.

## **Fast Gradient Sign Method (FGSM)**

The FGSM is a straightforward and effective method to generate adversarial examples. It works by computing the gradient of the loss with respect to the input data and perturbing the input in the direction that increases the loss the most.

### **Mathematical Formulation:**
Given an input image `x`, the model’s loss function `J`, and the model’s parameters `θ`, the adversarial example `x_adv` is generated as:

```math
x_adv = x + ε * sign(∇_x J(θ, x, y))
```

Where:
- `ε` is a small scalar value controlling the magnitude of the perturbation.
- `∇_x` is the gradient of the loss function with respect to the input image `x`.
- `sign()` returns the sign of the gradient, indicating the direction to perturb each pixel.

## **Model Used: MobileNetV2**

**MobileNetV2** is a lightweight convolutional neural network designed for mobile and embedded vision applications. It is widely used in real-time image classification tasks due to its efficiency and performance. This project uses MobileNetV2 pre-trained on the ImageNet dataset, allowing us to leverage a robust model for our experiments.

### **Advantages of MobileNetV2:**
- **Efficiency:** Optimized for devices with limited computational resources.
- **Versatility:** Performs well on a wide range of image classification tasks.

## **How the Code Works**

1. **Image Preprocessing:**
   - Images are resized to 224x224 pixels, normalized, and then fed into the MobileNetV2 model.
   
2. **Prediction:**
   - The model predicts the class of the original image and the adversarial image, displaying the most likely class label.
   
3. **Adversarial Example Generation:**
   - The FGSM method is used to generate the adversarial perturbations.
   
4. **Display Results:**
   - The original, perturbed, and adversarial images are displayed side by side, with labels showing the predicted classes.

## Potential Application

1. **Adversarial Perturbations for Identity Masking:**
   - Just as adversarial perturbations can be used to fool image classification models, similar techniques could be employed to alter biometric data (such as facial images) in a way that masks the original identity. For instance, small perturbations could be applied to a face image, causing a facial recognition system to misclassify or fail to recognize the person.

2. **Use in Protecting Privacy:**
   - This approach could help protect individuals' privacy by ensuring that sensitive biometric data, like facial images or fingerprints, cannot be easily linked to the real person by unauthorized systems. The perturbed images would appear almost identical to the original images to human observers but would be unrecognizable by AI systems.

3. **Applications in Data Sharing:**
   - When sharing datasets containing sensitive biometric information, adversarial perturbations could be applied to the data before sharing. This would allow researchers to work with the data without being able to identify the individuals, thus preserving privacy while still enabling useful analysis.

### **Challenges and Considerations:**

1. **Security and Reliability:**
   - While adversarial attacks can obscure identities, they can also be unstable. Minor changes in the system, such as retraining models or using different models, might make the perturbations ineffective. Ensuring consistent and reliable identity masking across different systems would be a key challenge.

2. **Ethical and Legal Implications:**
   - The use of adversarial techniques to mask identities must be carefully considered in terms of ethics and legality. There could be legal ramifications depending on how the data is used and shared, especially if it’s manipulated to deceive systems.

3. **Robustness Against Reverse Engineering:**
   - If someone knows that adversarial techniques are being used to mask identities, they might develop methods to reverse the perturbations and recover the original data. Ensuring the perturbations are robust against such reverse engineering attempts is crucial for maintaining privacy.

### **Modifications Needed:**

- **Tailoring Perturbations:** The perturbations would need to be specifically designed to target biometric recognition models rather than general image classifiers.
- **Testing Across Multiple Systems:** The effectiveness of the perturbations would need to be tested across various biometric recognition systems to ensure they generalize well and consistently mask identities.
- **Balancing Privacy and Usability:** The perturbations should be strong enough to mask identities but not so strong that they render the data unusable for legitimate purposes.

### **Potential Use Cases:**

- **Medical Data Sharing:** Masking identities in medical images for research purposes while ensuring patient confidentiality.
- **Anonymized Surveillance Data:** Using adversarial perturbations to anonymize individuals in surveillance footage before analysis.
- **Secure Online Authentication:** Protecting biometric data used in authentication systems from being exploited by third parties.

### **Conclusion:**
While this project provides a foundational understanding of adversarial attacks, extending it to mask identities in sensitive biometric data is both a promising and complex task. It requires careful consideration of technical, ethical, and legal challenges. The approach could provide a powerful tool for privacy preservation in a world where biometric data is increasingly used and shared.

## **Further Reading**

- **Adversarial Examples in Machine Learning:** [Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572)
- **MobileNetV2 Architecture:** [Sandler et al., 2018](https://arxiv.org/abs/1801.04381)
- **Robustness of Deep Learning Models:** [Madry et al., 2017](https://arxiv.org/abs/1706.06083)

## **Contributing**

Contributions are welcome! Feel free to submit a pull request or open an issue to discuss your ideas. Please follow the [contributing guidelines](CONTRIBUTING.md) when submitting code changes.
