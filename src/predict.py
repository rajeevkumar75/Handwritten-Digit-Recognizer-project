import argparse
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

def preprocess_image(image_path):
    """
    Loads an image, converts to grayscale 28x28, normalizes to [0,1],
    and ensures MNIST-like foreground/background (white digit on black).
    """
    img = Image.open(image_path).convert("L")  # grayscale
    # Make square by padding if needed
    w, h = img.size
    if w != h:
        size = max(w, h)
        new_img = Image.new("L", (size, size), color=255)  # white background
        new_img.paste(img, ((size - w)//2, (size - h)//2))
        img = new_img

    # Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to numpy and normalize
    arr = np.array(img).astype("float32")

    # If background is white (mean high), invert to MNIST style (white digit on black)
    # MNIST digits are white on black. If the mean > 127, likely black text on white -> invert.
    if arr.mean() > 127:
        arr = 255.0 - arr

    # Normalize to [0,1]
    arr = arr / 255.0

    # Expand dims to (1, 28, 28, 1)
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr

def main(args):
    model = tf.keras.models.load_model(args.model_path)
    x = preprocess_image(args.image_path)
    preds = model.predict(x, verbose=0)[0]
    pred_digit = int(np.argmax(preds))
    print(f"Predicted Digit: {pred_digit}")
    print("Class Probabilities:")
    for i, p in enumerate(preds):
        print(f"  {i}: {p*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict digit from image with trained MNIST model.")
    parser.add_argument("--model_path", type=str, default="models/mnist_cnn.h5")
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
