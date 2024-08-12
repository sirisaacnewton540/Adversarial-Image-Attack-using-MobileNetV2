import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

# Function to load and preprocess an image
def load_and_preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

# Function to decode the predictions
def decode_predictions(pred):
    return tf.keras.applications.mobilenet_v2.decode_predictions(pred, top=1)[0]

# FGSM for generating adversarial examples
def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = tf.keras.losses.categorical_crossentropy(input_label, prediction)
    
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

# Function to process the image and show results
def process_image(img_path):
    # Load and preprocess the image
    image = load_and_preprocess_image(img_path)
    image = tf.convert_to_tensor(image)

    # Get the original prediction
    prediction = model.predict(image)
    decoded_preds = decode_predictions(prediction)
    original_class = decoded_preds[0][1]
    print(decoded_preds)

    # Get the label of the highest confidence prediction
    label = np.argmax(prediction)
    label = tf.one_hot(label, prediction.shape[-1])
    label = tf.reshape(label, (1, prediction.shape[-1]))

    # Create adversarial perturbations
    perturbations = create_adversarial_pattern(image, label)
    epsilon = 0.01
    adversarial_image = image + epsilon * perturbations
    adversarial_image = tf.clip_by_value(adversarial_image, -1, 1)
    prediction1 = model.predict(adversarial_image)
    decoded_preds1 = decode_predictions(prediction1)
    adv_class = decoded_preds1[0][1]
    print(decoded_preds1)

    # Convert images to displayable format
    original_img = tf.keras.preprocessing.image.array_to_img(image[0])
    perturbation_img = tf.keras.preprocessing.image.array_to_img(0.5 * perturbations[0] + 0.5)
    adversarial_img = tf.keras.preprocessing.image.array_to_img(adversarial_image[0])

    # Add class labels to each image
    draw_text_on_image(original_img, f"[Original Class]: {original_class}")
    draw_text_on_image(adversarial_img, f"[Adversarial Class]: {adv_class}")
    draw_text_on_image(perturbation_img, "[Perturbations]")

    # Combine and display images
    display_image(original_img, perturbation_img, adversarial_img)

# Function to draw text on an image
def draw_text_on_image(image, text):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    # Calculate text size with the correct method
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
    # Add a rectangle behind the text for contrast
    draw.rectangle(((0, 0), (text_size[0] + 10, text_size[1] + 10)), fill="black")
    draw.text((5, 5), text, font=font, fill="white")

# Function to display the images using tkinter
def display_image(original_img, perturbation_img, adversarial_img):
    result_window = tk.Toplevel(root)
    result_window.title("Adversarial Attack Results")

    # Combine images into a single image for display
    combined_image = Image.new('RGB', (original_img.width * 3, original_img.height))
    combined_image.paste(original_img, (0, 0))
    combined_image.paste(perturbation_img, (original_img.width, 0))
    combined_image.paste(adversarial_img, (original_img.width * 2, 0))

    # Display the combined image in the GUI
    combined_image_tk = ImageTk.PhotoImage(combined_image)
    label = tk.Label(result_window, image=combined_image_tk)
    label.image = combined_image_tk  # Keep a reference to avoid garbage collection
    label.pack()

# Function to open a file dialog and load an image
def load_image():
    img_path = filedialog.askopenfilename()
    if img_path:
        process_image(img_path)

# Create the main tkinter window
root = tk.Tk()
root.title("Adversarial Image Generator")

# Add a button to load the image
load_button = tk.Button(root, text="Load Image", command=load_image, font=('Arial', 20))
load_button.pack()

# Run the GUI event loop
root.mainloop()
