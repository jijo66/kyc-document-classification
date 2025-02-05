
# Automated KYC Document Classification

## Overview
This project focuses on automating the classification of Know Your Customer (KYC) documents—specifically Kenyan identity cards and passports—using deep learning techniques. The primary goal is to enhance the efficiency of document verification processes in the banking and fintech sectors in Kenya, reducing the reliance on manual verification methods.

## Objectives
- Develop a deep learning model to classify KYC documents.
- Evaluate the model's performance using various metrics.
- Implement a user-friendly web interface for document submission and classification.

## Technologies Used
- **Python**: The primary programming language for implementation.
- **TensorFlow/Keras**: Libraries used for building and training the deep learning model.
- **Streamlit**: A framework for creating the web interface.
- **NumPy** and **Pandas**: Libraries for data manipulation and numerical computations.
- **Matplotlib**: Used for visualizing training results and performance metrics.

## Dataset
The dataset consists of approximately 150 images of Kenyan KYC documents, including both identity cards and passports. The images were sourced from publicly available datasets and are organized into training, validation, and testing subsets.

### Data Preprocessing
Before training the model, several preprocessing steps were performed:
- Images were resized to a uniform dimension of 180x180 pixels.
- Pixel values were normalized to a range of [0, 1].
- Data augmentation techniques were applied to increase dataset variability.

## Model Architecture
The model is built using a Sequential Convolutional Neural Network (CNN) architecture, which includes:
- Convolutional layers for feature extraction.
- Max pooling layers to reduce dimensionality.
- Dropout layers to prevent overfitting.
- A fully connected layer with a Softmax activation function for multi-class classification.

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/KYC-Document-Classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd KYC-Document-Classification
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit web application:
   ```bash
   streamlit run app.py
   ```

## Usage
Once the application is running, users can upload KYC documents (IDs or passports) through the web interface. The model will classify the documents and display the predicted class along with confidence levels.

## Key Findings
The trained model achieved high accuracy in classifying KYC documents, demonstrating its effectiveness in automating verification processes. The implementation of a web interface enhances user interaction and facilitates quicker document processing.

## Future Work
Future research may focus on optimizing model performance further, expanding the dataset for better generalization, integrating with existing banking systems, and conducting user experience studies on the web interface.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
