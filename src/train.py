import argparse
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Reproducibility (optional)
tf.random.set_seed(42)
np.random.seed(42)

def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main(args):
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0

    # Reshape to (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)

    # Build model
    model = build_model()

    # Callbacks
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    ckpt_path = os.path.join(os.path.dirname(args.model_path), "best_weights.keras")
    cbs = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, save_weights_only=False)
    ]

    # Train
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cbs,
        verbose=2
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Save final model
    model.save(args.model_path)
    print(f"Saved model to: {args.model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_path", type=str, default="models/mnist_cnn.h5")
    args = parser.parse_args()
    main(args)
