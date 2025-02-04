# 🌱 Plant Disease Detection Application

A user-friendly web application built with Streamlit that detects plant diseases from leaf images using deep learning. The app provides detailed analysis, confidence levels, and tailored agricultural guidance for plant disease treatment and prevention.

---

## 📋 Features

1. **Image Upload & Preview**
   - Drag and drop or browse to upload plant leaf images
   - Supports multiple image formats including JPG, PNG, and JPEG
   - Real-time image preview with intuitive user interface

2. **Disease Detection**
   - Advanced deep learning model for accurate plant disease prediction
   - Displays comprehensive diagnostic information
   - Shows precise confidence percentage for each prediction

3. **Detailed Analysis**
   - Interactive visual representation of prediction confidence
   - Displays top disease probabilities with clear graphical breakdown
   - Provides in-depth insights into potential plant health issues

4. **Agricultural Guidance**
   - Comprehensive treatment recommendations for detected diseases
   - Tailored prevention strategies based on specific plant conditions
   - Expert-curated best practices for maintaining crop health

5. **Customizable UI**
   - Sleek, modern design with responsive CSS styling
   - Intuitive user experience across different devices
   - Clean and professional interface for easy navigation

---

## 🚀 Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Git (optional, but recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Machine Learning Resources
1. Download the pre-trained model:
   - `plant_disease_model.h5`
   - `class_indices.json`
2. Place these files in the project's root directory

### Step 5: Launch the Application
```bash
streamlit run app.py
```

### Step 6: Access the Web Application
Open your web browser and navigate to:
```
http://localhost:8501
```

---

## 🖥️ Usage Guide

1. **Image Upload**
   - Click on the upload area or drag and drop a leaf image
   - Supported formats: JPG, PNG, JPEG
   - Ensure the image is clear and focuses on the entire leaf

2. **Disease Detection**
   - Wait for the AI model to analyze the uploaded image
   - Review the predicted disease and confidence score
   - Examine the detailed analysis section

3. **Exploring Recommendations**
   - Read through treatment and prevention strategies
   - Take note of specific care instructions
   - Consult local agricultural experts for personalized advice

---

## 📚 Technologies Used

- **Frontend**: Streamlit
- **Machine Learning**: TensorFlow, Keras
- **Data Visualization**: Matplotlib, Plotly
- **Image Processing**: OpenCV, PIL
- **Languages**: Python

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` file for more information.

---

## 💡 Future Roadmap

- [ ] Expand disease detection database
- [ ] Implement multi-language support
- [ ] Add real-time camera detection
- [ ] Develop mobile application
- [ ] Integrate cloud deployment

---

## 🌿 About the Project

This project aims to democratize agricultural technology by providing farmers and gardeners with an accessible, AI-powered tool for plant disease detection. By leveraging cutting-edge machine learning, we strive to support sustainable agriculture and crop health management.

---

## 🙏 Acknowledgments

- Open-source community
- Streamlit Team
- Machine Learning researchers
- Agricultural experts and advisors

---

🌍 **Empowering Agriculture through Artificial Intelligence** 🌍