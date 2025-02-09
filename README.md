﻿# 🌱Plant Disease Detector 🔍 with Streamlit

This project uses a Convolutional Neural Network (CNN) to detect plant diseases from images. The project is divided into two main parts:

1. **Model Training and Evaluation:**
   This section covers dataset preparation, data preprocessing, CNN model design and training, evaluation, and saving the model in the `.h5` format.
   *For the full implementation, please refer to the accompanying Jupyter Notebook.*

2. **Streamlit Integration and Cloud Deployment:**
   This section explains how to integrate the trained model into a Streamlit web application and deploy the app on a cloud using the same Streamlit Platform.

---

## Part 1: Model Training and Evaluation

The following outlines the workflow for training and evaluating the CNN model on the Plant Village dataset:

### 1. Dataset Preparation

- **Dataset Structure:**
  Organize your dataset into three main folders: `Train`, `Val`, and `Test`, with each containing sub-folders for each of the 15 classes. For example:

  ```
  Train/
     ├── Class_1/
     ├── Class_2/
     └── ... (15 classes)
  Val/
     ├── Class_1/
     ├── Class_2/
     └── ... (15 classes)
  Test/
     ├── Class_1/
     ├── Class_2/
     └── ... (15 classes)
  ```

- **Data Preprocessing:**
  - **Resize Images:** All images are resized to 256x256 pixels.
  - **Normalization:** Pixel values are scaled to the range [0, 1].
  - **Data Augmentation:** For training data, augmentation techniques such as rotation, horizontal flipping, zooming, and shearing are applied to improve model robustness.
  - **Validation/Test Preprocessing:** Only normalization is applied to maintain the integrity of evaluation data.

### 2. Model Architecture

- **CNN Design:**
  The model replicates the architecture described in the referenced paper. It includes:
  - Several convolutional layers with increasing filter sizes.
  - Max pooling layers to reduce spatial dimensions.
  - Fully connected (dense) layers, with the final layer modified to output predictions across 15 classes.

### 3. Model Compilation

- **Compilation Settings:**
  - **Optimizer:** Adam with a learning rate of 0.0001
  - **Loss Function:** Categorical Cross-Entropy
  - **Metrics:** Accuracy

### 4. Model Training

- **Training Process:**
  - The model is trained for 15 epochs.
  - Early stopping is used to prevent overfitting by restoring the best model weights when performance plateaus.

### 5. Model Evaluation

- **Evaluation Metrics:**
  The trained model is evaluated on the test set using accuracy, precision, recall, and F1-score.
- **Visualization:**
  A confusion matrix is generated to provide a detailed breakdown of predictions for each class.

### 6. Saving the Model

- **Output:**
  The final model is saved in the `.h5` format, which can be later loaded into the Streamlit application for inference.

> **Note:** The complete code implementation and detailed steps are available in the accompanying Jupyter Notebook.

---

## Part 2: Streamlit Integration and Cloud Deployment

This part describes how to create a user-friendly web application using Streamlit, integrate the pre-trained model, and deploy the application on the cloud.

### Streamlit App Development

- **User Interface:**
  - Build an intuitive interface that allows users to upload plant images.
  - Display the predicted disease along with the model’s confidence scores.

- **Model Integration:**
  - Load the saved `.h5` model within the Streamlit app.
  - Preprocess incoming images (resizing, normalization, etc.) consistent with the training pipeline.
  - Run the prediction and display the results directly on the app.

- **Visualization:**
  - Optionally, include visual elements (e.g., prediction probability charts) to enhance user understanding.

### Cloud Deployment

- **Choosing a Platform:**
  Platforms like **Streamlit Cloud**, **Heroku**, **AWS**, or others can be used for hosting the app.

- **Deployment Steps:**
  1. **Environment Setup:** Ensure all dependencies (Python libraries, TensorFlow, etc.) are installed on the cloud server.
  2. **Code Push:** Deploy your application code to the chosen platform (e.g., via GitHub integration).
  3. **Configuration:** Set up necessary environment variables and configurations specific to your cloud provider.
  4. **Launch:** Start your Streamlit app and share the live URL with end users.

> **Tip:** For detailed deployment instructions, refer to the official [Streamlit documentation](https://docs.streamlit.io/) and your chosen cloud provider’s guides.

---

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/plant-disease-detection.git
   cd plant-disease-detection
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model:**
   - Open and run the Jupyter Notebook provided in the repository to prepare your dataset, train, and evaluate the model.

4. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

5. **Deployment:**
   - Follow the cloud deployment steps as outlined above to host your app online.

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- Appreciation goes to the authors of the referenced paper for their work on the model architecture.
- Thanks to the contributors of the Plant Village dataset for providing high-quality data.

---

This README provides a concise overview of both the model training process and the subsequent steps for deploying the application with Streamlit. Adjust the content as needed for your project specifics.
