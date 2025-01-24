import streamlit as st
import numpy as np
from PIL import Image
import json
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Custom CSS styling
def inject_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
            
            * {
                font-family: 'Inter', sans-serif;
            }
            
            .main {
                background-color: #f8fafc;
            }
            
            .header {
                text-align: center;
                padding: 2rem 0;
                background: linear-gradient(90deg, #059669 0%, #10b981 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .upload-box {
                background: white;
                border-radius: 12px;
                padding: 2rem;
                margin: 2rem 0;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                border: 1px solid #e2e8f0;
            }
            
            .result-card {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                border-left: 4px solid #10b981;
            }
            
            .image-preview {
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            
            .stProgress > div > div > div > div {
                background-color: #10b981;
            }
        </style>
    """, unsafe_allow_html=True)

# Load model and class indices
@st.cache_resource
def load_model_and_indices():
    model = load_model('plant_disease_model.h5')
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    return model, class_indices

# Image preprocessing function
def load_and_preprocess_image(image, target_size=(224, 224)):
    try:
        img = image.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.
        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

# Prediction function
def predict_image_class(model, image, class_indices):
    try:
        preprocessed_img = load_and_preprocess_image(image)
        predictions = model.predict(preprocessed_img)
        probabilities = predictions[0]
        sorted_indices = sorted(class_indices.keys(), key=lambda x: int(x))
        class_probs = [(class_indices[idx], probabilities[int(idx)] * 100) for idx in sorted_indices]
        class_probs.sort(key=lambda x: x[1], reverse=True)
        return class_probs[0][0], class_probs[0][1], class_probs
    except Exception as e:
        raise ValueError(f"Error making prediction: {e}")

# Disease recommendations database
disease_recommendations = {
    'Tomato Early Blight': {
        'description': 'A fungal disease causing concentric rings on leaves',
        'treatment': [
            'Apply copper-based fungicides',
            'Remove infected leaves',
            'Use chlorothalonil sprays'
        ],
        'prevention': [
            'Practice crop rotation',
            'Improve air circulation',
            'Water at plant base'
        ]
    },
    'Potato Late Blight': {
        'description': 'Destructive fungal disease affecting potatoes',
        'treatment': [
            'Apply mancozeb fungicides',
            'Destroy infected plants',
            'Avoid overhead watering'
        ],
        'prevention': [
            'Use certified seed potatoes',
            'Monitor humidity levels',
            'Apply preventative sprays'
        ]
    },
    'Corn Common Rust': {
        'description': 'Fungal disease with reddish-brown pustules',
        'treatment': [
            'Apply sulfur-based treatments',
            'Remove affected leaves',
            'Use fungicidal sprays'
        ],
        'prevention': [
            'Plant resistant hybrids',
            'Avoid late planting',
            'Balance soil nutrients'
        ]
    }
}

# Streamlit app
def main():
    inject_custom_css()
    
    st.markdown("""
        <div class="header">
            <h1 style='font-size: 2.5rem; margin: 0;'>Plant Disease Detection</h1>
            <p style='color: #64748b; margin: 0.5rem 0;'>Upload a plant leaf image to detect potential diseases</p>
        </div>
    """, unsafe_allow_html=True)

    try:
        model, class_indices = load_model_and_indices()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return


    with st.container():
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop or browse files",
            type=["jpg", "png", "jpeg"],
            help="Supported formats: JPG, PNG, JPEG",
            key="uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner('Analyzing image...'):
            image = Image.open(uploaded_file)
            st.markdown('<p style="color: #64748b; margin-bottom: 0.5rem;">Image Preview</p>', unsafe_allow_html=True)
            st.markdown('<div class="image-preview">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            try:
                prediction, confidence, all_probs = predict_image_class(model, image, class_indices)
                st.markdown(f"""
                    <div class="result-card">
                        <h3 style="margin: 0 0 1rem 0; color: #1e293b;">Analysis Results</h3>
                        <div style="display: flex; gap: 2rem;">
                            <div>
                                <p style="color: #64748b; margin: 0;">Diagnosis</p>
                                <p style="color: #1e293b; margin: 0; font-size: 1.2rem; font-weight: 600;">{prediction}</p>
                            </div>
                            <div>
                                <p style="color: #64748b; margin: 0;">Confidence</p>
                                <p style="color: #1e293b; margin: 0; font-size: 1.2rem; font-weight: 600;">{confidence:.2f}%</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📊 View Detailed Analysis & Recommendations"):
                    # Visualization
                    st.subheader("Prediction Confidence Distribution")
                    top_n = 5
                    top_probs = all_probs[:top_n]
                    
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.barh(
                        [x[0] for x in top_probs], 
                        [x[1] for x in top_probs],
                        color='#10b981'
                    )
                    ax.set_xlabel('Confidence (%)')
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Recommendations
                    st.subheader("Agricultural Guidance")
                    
                    if prediction in disease_recommendations:
                        info = disease_recommendations[prediction]
                        st.markdown(f"**{prediction} Overview**: {info['description']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**🩹 Treatment Options**")
                            for treatment in info['treatment']:
                                st.markdown(f"- {treatment}")
                        
                        with col2:
                            st.markdown("**🛡️ Prevention Strategies**")
                            for prevention in info['prevention']:
                                st.markdown(f"- {prevention}")
                    else:
                        st.info("General Best Practices:")
                        st.markdown("- Rotate crops regularly to prevent soil depletion")
                        st.markdown("- Monitor plants weekly for early signs of stress")
                        st.markdown("- Maintain proper spacing between plants")
                        st.markdown("- Use organic mulch to regulate soil moisture")
                        st.markdown("- Test soil nutrients annually and amend as needed")

                    st.markdown("""
                        <div style="margin-top: 1rem; padding: 1rem; background-color: #f0fdf4; border-radius: 8px;">
                            <small>Note: Recommendations are general guidelines. Always consult with local agricultural experts for specific advice.</small>
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                    <div class="result-card" style="border-color: #ef4444;">
                        <h3 style="margin: 0 0 1rem 0; color: #1e293b;">Error</h3>
                        <p style="color: #ef4444; margin: 0;">{str(e)}</p>
                    </div>
                """, unsafe_allow_html=True)

    
    # Add this section right after the header in the main() function
    st.markdown("""
        <div style="margin: 1.5rem 0 2rem 0;">
            <h3 style="color: #1e293b; margin-bottom: 1rem;">🌱 Currently Supported Plants</h3>
            <div style="display: grid; 
                        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
                        gap: 0.8rem;
                        padding: 0.5rem;">
                <div style="background: #f0fdf4; 
                            padding: 0.6rem; 
                            border-radius: 8px; 
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🍎 Apple
                </div>
                <div style="background: #f0fdf4; 
                            padding: 0.6rem; 
                            border-radius: 8px; 
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🔵 Blueberry
                </div>
                <div style="background: #f0fdf4; 
                            padding: 0.6rem; 
                            border-radius: 8px; 
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🍒 Cherry
                </div>
                <div style="background: #f0fdf4; 
                            padding: 0.6rem; 
                            border-radius: 8px; 
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🌽 Corn
                </div>
                <div style="background: #f0fdf4; 
                            padding: 0.6rem; 
                            border-radius: 8px; 
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🍇 Grape
                </div>
                <div style="background: #f0fdf4; 
                            padding: 0.6rem; 
                            border-radius: 8px; 
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🍊 Orange
                </div>
                <div style="background: #f0fdf4; 
                            padding: 0.6rem; 
                            border-radius: 8px; 
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🍑 Peach
                </div>
                <div style="background: #f0fdf4; 
                            padding: 0.6rem; 
                            border-radius: 8px; 
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🌶️ Pepper
                </div>
                <div style="background: #f0fdf4; 
                            padding: 0.6rem; 
                            border-radius: 8px; 
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🥔 Potato
                </div>
                <div style="background: #f0fdf4; 
                            padding: 0.6rem; 
                            border-radius: 8px; 
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🍓 Raspberry
                </div>
                <div style="background: #f0fdf4; 
                            padding: 0.6rem; 
                            border-radius: 8px; 
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🌱 Soybean
                </div>
                <div style="background: #f0fdf4; 
                            padding: 0.6rem; 
                            border-radius: 8px; 
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🎃 Squash
                </div>
                <div style="background: #f0fdf4; 
                            padding: 0.6rem; 
                            border-radius: 8px; 
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🍓 Strawberry
                </div>
                <div style="background: #f0fdf4; 
                            padding: 0.6rem; 
                            border-radius: 8px; 
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🍅 Tomato
                </div>
            </div>
            <p style="color: #64748b; margin-top: 0.8rem; font-size: 0.9rem;">
                More plant types coming soon!
            </p>
        </div>
    """, unsafe_allow_html=True)




    st.markdown("""
        <div style="text-align: center; margin-top: 3rem; color: #64748b;">
            <p>🌿 Healthy plants, happier planet</p>
            <p style="font-size: 0.9rem;">Upload clear images of leaves for best results</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()