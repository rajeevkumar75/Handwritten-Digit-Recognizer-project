# Handwritten Digit Recognizer (TensorFlow/Keras)

Link: https://rajeevkumar75-handwritten-digit-recognizer-project-app-uxx0ad.streamlit.app/

A complete, beginner-friendly project to train a CNN on the MNIST dataset and predict digits (0–9).  
It includes:
- Training script (`src/train.py`)
- Prediction script (`src/predict.py`)
- Streamlit app for interactive prediction (`app.py`)
- Requirements file (`requirements.txt`)

Summary: 
This project involves using a convolutional neural network (CNN) built with TensorFlow to 
classify digits (0–9) from the MNIST dataset. The model will be trained on thousands of labeled 
images and used to predict digits in new image samples. 


Description: 
• Load the MNIST dataset using TensorFlow/Keras 
• Preprocess the images (normalize pixel values, reshape input) 
• Define a CNN architecture with convolution, pooling, and dense layers 
• Compile and train the model using the training set 
• Evaluate performance on the test set using accuracy and loss metrics 
• Predict digits from unseen images and display the output 


Functional Components: 
• Import TensorFlow and load the MNIST dataset 
• Preprocess and reshape image data 
• Build CNN model with Conv2D, MaxPooling2D, Flatten, Dense layers 
• Train model with fit() and validate on test data 
• Predict using model.predict() 
• Visualize predictions using matplotlib (optional) 


Sample Dataset (MNIST format): 
Each input: 28×28 grayscale image of a handwritten digit 
Sample label: 7 
Pixel values range: 0 to 255 


Expected Output: 
For an input image resembling the digit 7: 
Predicted Digit: 7 
Model Accuracy: 98 percent on test data 
