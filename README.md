# Handwritten Digit Recognizer (TensorFlow/Keras)

A complete, beginner-friendly project to train a CNN on the MNIST dataset and predict digits (0–9).  
It includes:
- Training script (`src/train.py`)
- Prediction script (`src/predict.py`)
- Streamlit app for interactive prediction (`app.py`)
- Requirements file (`requirements.txt`)

## 1) Quick Start (Google Colab)
1. Open Google Colab.
2. Upload the following files from this project:
   - `src/train.py`
   - `src/predict.py`
   - `app.py`
   - `requirements.txt`
3. Install dependencies (if needed):
```python
!pip -q install -r requirements.txt
```
4. Train:
```python
!python src/train.py --epochs 10 --batch_size 128 --model_path models/mnist_cnn.h5
```
5. Predict on your image:
```python
!python src/predict.py --model_path models/mnist_cnn.h5 --image_path sample_7.png
```
6. Run Streamlit app (in local Jupyter/VS Code):
```bash
streamlit run app.py
```

## 2) Local Setup (Windows/Mac/Linux)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Train
python src/train.py --epochs 10 --batch_size 128 --model_path models/mnist_cnn.h5

# Predict
python src/predict.py --model_path models/mnist_cnn.h5 --image_path path/to/your_digit.png
```

## 3) Notes on Images
- The model expects **28x28 grayscale** images with **white digit on black background** (MNIST style).
- The prediction script auto-detects common cases (black on white) and inverts if needed.
- You can provide any small square image; it will be resized, converted to grayscale, and normalized.

## 4) Files
```
handwritten_digit_recognizer/
├── app.py
├── requirements.txt
├── src/
│   ├── train.py
│   └── predict.py
├── models/           # saved models go here (created at runtime)
└── notebooks/        # optional space for your own notebooks
```

## 5) Expected Results
- Test accuracy around **98%** after ~10 epochs on CPU (varies by run).
- `predict.py` prints the predicted digit and a probability table.

## 6) Troubleshooting
- If TensorFlow install fails on older CPUs, try: `pip install tensorflow-cpu>=2.12,<3.0`.
- If memory is low, reduce `--batch_size` (e.g., 64 or 32).
- On Apple Silicon (M1/M2), you can install `tensorflow-macos` and `tensorflow-metal` for acceleration.
"# Handwritten-Digit-Recognizer-" 
