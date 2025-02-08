import streamlit as st
import numpy as np
from PIL import Image
import json
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Custom CSS Styling for a Professional Look and Feel
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Load Model and Class Indices (Cached for Performance)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_and_indices():
    model = load_model('plant_disease_cnn.h5')
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    return model, class_indices


# ---------------------------------------------------------------------------
# Image Preprocessing Function
# ---------------------------------------------------------------------------
def load_and_preprocess_image(image, target_size=(256, 256)):
    try:
        img = image.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.
        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")


# ---------------------------------------------------------------------------
# Prediction Function with Confidence Threshold
# ---------------------------------------------------------------------------
def predict_image_class(model, image, class_indices, confidence_threshold=70.0):
    try:
        preprocessed_img = load_and_preprocess_image(image)
        predictions = model.predict(preprocessed_img)
        probabilities = predictions[0]
        sorted_indices = sorted(class_indices.keys(), key=lambda x: int(x))
        class_probs = [(class_indices[idx], probabilities[int(idx)] * 100) for idx in sorted_indices]
        class_probs.sort(key=lambda x: x[1], reverse=True)

        # Check confidence against threshold
        if class_probs[0][1] < confidence_threshold:
            return 'Unknown', class_probs[0][1], class_probs
        return class_probs[0][0], class_probs[0][1], class_probs
    except Exception as e:
        raise ValueError(f"Error making prediction: {e}")

# ---------------------------------------------------------------------------
# Disease Recommendations Database
# ---------------------------------------------------------------------------
disease_recommendations = {
    "Apple_Black_Rot": {
         "description": "Apple Black Rot is a fungal disease that causes dark, circular lesions on leaves and fruit.",
         "treatment": [
             "Remove infected leaves and fruit",
             "Apply appropriate fungicides",
             "Ensure proper pruning to improve airflow"
         ],
         "prevention": [
             "Regularly monitor trees",
             "Practice proper spacing",
             "Maintain garden sanitation"
         ]
    },
    "Apple_Healthy": {
         "description": "The apple appears healthy.",
         "treatment": [],
         "prevention": [
             "Continue with standard cultural practices"
         ]
    },
    "Apple_Scab": {
         "description": "Apple Scab is a fungal disease that causes dark, scabby lesions on leaves and fruit.",
         "treatment": [
             "Apply fungicides during early infection",
             "Remove fallen leaves and fruit"
         ],
         "prevention": [
             "Use resistant varieties",
             "Ensure good air circulation"
         ]
    },
    "Bell_Pepper_Healthy": {
         "description": "The bell pepper appears healthy.",
         "treatment": [],
         "prevention": [
             "Maintain proper irrigation and balanced fertilization"
         ]
    },
    "Bell_Pepper_Bacterial_Spot": {
         "description": "Bacterial Spot causes small dark spots on leaves and fruit of bell peppers.",
         "treatment": [
             "Remove infected parts",
             "Apply copper-based bactericides"
         ],
         "prevention": [
             "Avoid overhead watering",
             "Practice crop rotation"
         ]
    },
    "Cedar_Apple_Rust": {
         "description": "Cedar Apple Rust is a fungal disease that requires both cedar and apple hosts for its lifecycle.",
         "treatment": [
             "Prune infected branches",
             "Remove nearby alternate hosts if possible"
         ],
         "prevention": [
             "Plant resistant apple varieties",
             "Monitor for early symptoms"
         ]
    },
    "Cherry_Healthy": {
         "description": "The cherry appears healthy.",
         "treatment": [],
         "prevention": [
             "Maintain good cultural practices and regular inspection"
         ]
    },
    "Cherry_Powdery_Mildew": {
         "description": "Powdery Mildew causes a white powdery coating on cherry leaves.",
         "treatment": [
             "Apply sulfur-based fungicides",
             "Remove affected leaves"
         ],
         "prevention": [
             "Ensure proper air circulation",
             "Avoid excessive nitrogen fertilization"
         ]
    },
    "Grape_Black_Rot": {
         "description": "Black Rot is a fungal disease in grapes causing dark, rotted lesions on leaves and fruit.",
         "treatment": [
             "Remove infected parts",
             "Apply appropriate fungicides"
         ],
         "prevention": [
             "Prune regularly",
             "Maintain good vineyard hygiene"
         ]
    },
    "Grape_Esca_(Black_Measles)": {
         "description": "Esca (Black Measles) is a grapevine disease affecting the trunk and leaves.",
         "treatment": [
             "Remove severely affected vines",
             "Consult local guidelines for fungicide use"
         ],
         "prevention": [
             "Avoid injury to vines",
             "Maintain proper vineyard management"
         ]
    },
    "Grape_Healthy": {
         "description": "The grapevine appears healthy.",
         "treatment": [],
         "prevention": [
             "Regular inspection and maintenance"
         ]
    },
    "Grape_Leaf_Blight": {
         "description": "Leaf Blight causes necrotic lesions on grape leaves.",
         "treatment": [
             "Remove infected leaves",
             "Apply fungicides to limit spread"
         ],
         "prevention": [
             "Ensure proper air circulation",
             "Practice regular pruning"
         ]
    },
    "Maize_Cercospora_Leaf_Spot": {
         "description": "Cercospora Leaf Spot causes small, circular spots on maize leaves.",
         "treatment": [
             "Apply fungicides when necessary",
             "Remove crop residues after harvest"
         ],
         "prevention": [
             "Practice crop rotation",
             "Use resistant varieties if available"
         ]
    },
    "Maize_Common_Rust": {
         "description": "Common Rust presents as rust-colored pustules on maize leaves.",
         "treatment": [
             "Apply fungicides early in the season",
             "Remove infected leaves"
         ],
         "prevention": [
             "Plant resistant hybrids",
             "Rotate crops annually"
         ]
    },
    "Maize_Healthy": {
         "description": "The maize crop appears healthy.",
         "treatment": [],
         "prevention": [
             "Continue with proper field management practices"
         ]
    },
    "Maize_Northern_Leaf_Blight": {
         "description": "Northern Leaf Blight causes elongated, grayish lesions on maize leaves.",
         "treatment": [
             "Apply fungicides during high disease pressure",
             "Remove infected debris from the field"
         ],
         "prevention": [
             "Practice crop rotation",
             "Use resistant hybrids"
         ]
    },
    "Peach_Bacterial_Spot": {
         "description": "Bacterial Spot causes dark, water-soaked lesions on peach leaves and fruit.",
         "treatment": [
             "Apply copper-based bactericides",
             "Prune infected branches"
         ],
         "prevention": [
             "Avoid overhead irrigation",
             "Ensure proper spacing for air flow"
         ]
    },
    "Peach_Healthy": {
         "description": "The peach appears healthy.",
         "treatment": [],
         "prevention": [
             "Maintain optimal growing conditions"
         ]
    },
    "Potato_Early_Blight": {
         "description": "Early Blight in potatoes shows as small, brown spots (often with concentric rings) on leaves.",
         "treatment": [
             "Apply recommended fungicides",
             "Remove and destroy infected foliage"
         ],
         "prevention": [
             "Rotate crops to reduce pathogen buildup",
             "Ensure adequate plant spacing"
         ]
    },
    "Potato_Healthy": {
         "description": "The potato crop appears healthy.",
         "treatment": [],
         "prevention": [
             "Continue with regular care and monitoring"
         ]
    },
    "Potato_Late_Blight": {
         "description": "Late Blight causes large, dark lesions and rapid decay in potato plants.",
         "treatment": [
             "Remove and destroy infected plants immediately",
             "Apply fungicides as per local recommendations"
         ],
         "prevention": [
             "Use resistant varieties if available",
             "Ensure proper field drainage"
         ]
    },
    "Strawberry_Healthy": {
         "description": "The strawberry plant appears healthy.",
         "treatment": [],
         "prevention": [
             "Maintain proper spacing and good hygiene"
         ]
    },
    "Strawberry_Leaf_Scorch": {
         "description": "Leaf Scorch in strawberries causes brown edges on the leaves.",
         "treatment": [
             "Adjust irrigation practices",
             "Remove severely scorched leaves"
         ],
         "prevention": [
             "Avoid water stress",
             "Use mulch to regulate soil moisture"
         ]
    },
    "Tomato_Bacterial_Spot": {
         "description": "Bacterial Spot results in small dark lesions on tomato leaves and fruits.",
         "treatment": [
             "Apply copper-based bactericides",
             "Remove and discard infected parts"
         ],
         "prevention": [
             "Ensure good air circulation",
             "Practice crop rotation"
         ]
    },
    "Tomato_Early_Blight": {
         "description": "Early Blight appears as circular spots with concentric rings on tomato leaves.",
         "treatment": [
             "Apply appropriate fungicides",
             "Remove infected plant debris"
         ],
         "prevention": [
             "Plant resistant varieties",
             "Maintain garden hygiene"
         ]
    },
    "Tomato_Healthy": {
         "description": "The tomato plant appears healthy.",
         "treatment": [],
         "prevention": [
             "Continue with proper care and regular inspection"
         ]
    },
    "Tomato_Late_Blight": {
         "description": "Late Blight causes rapid decay with dark lesions on tomato plants.",
         "treatment": [
             "Remove and destroy infected plants",
             "Apply fungicides immediately"
         ],
         "prevention": [
             "Avoid overhead watering",
             "Ensure proper plant spacing for airflow"
         ]
    },
    "Tomato_Septoria_Leaf_Spot": {
         "description": "Septoria Leaf Spot appears as small, dark spots on tomato leaves that may merge.",
         "treatment": [
             "Apply fungicides targeting septoria",
             "Remove infected leaves"
         ],
         "prevention": [
             "Maintain garden cleanliness",
             "Practice crop rotation"
         ]
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
         "description": "Tomato Yellow Leaf Curl Virus causes yellowing and curling of leaves, reducing yield.",
         "treatment": [
             "Remove infected plants",
             "Control insect vectors (e.g., whiteflies)"
         ],
         "prevention": [
             "Use resistant tomato varieties if available",
             "Implement strict sanitation measures"
         ]
    }
}


# ---------------------------------------------------------------------------
# Main Application Function
# ---------------------------------------------------------------------------
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
        uploaded_file = st.file_uploader(
            "Drag and drop or browse files",
            type=["jpg", "png", "jpeg"],
            help="Supported formats: JPG, PNG, JPEG",
            key="uploader"
        )

    if uploaded_file is not None:
        with st.spinner('Analyzing image...'):
            image = Image.open(uploaded_file)
            st.markdown('<p style="color: #64748b; margin-bottom: 0.5rem;">Image Preview</p>', unsafe_allow_html=True)
            st.markdown('<div class="image-preview">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            try:
                prediction, confidence, all_probs = predict_image_class(model, image, class_indices)
                result_color = "#10b981" if prediction != "Unknown" else "#ef4444"

                st.markdown(f"""
                    <div class="result-card" style="border-color: {result_color};">
                        <h3 style="margin: 0 0 1rem 0; color: #1e293b;">Analysis Results</h3>
                        <div style="display: flex; gap: 2rem;">
                            <div>
                                <p style="color: #64748b; margin: 0;">Diagnosis</p>
                                <p style="color: {result_color}; margin: 0; font-size: 1.2rem; font-weight: 600;">
                                    {prediction if prediction != 'Unknown' else 'Non-plant/Unknown'}
                                </p>
                            </div>
                            <div>
                                <p style="color: #64748b; margin: 0;">Confidence</p>
                                <p style="color: {result_color}; margin: 0; font-size: 1.2rem; font-weight: 600;">
                                    {confidence:.2f}%{'' if prediction != 'Unknown' else ' (Below threshold)'}
                                </p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                if prediction == "Unknown":
                    st.warning("""
                        This image doesn't appear to show a recognizable plant leaf.
                        Please upload a clear image of a plant leaf for disease detection.
                    """)
                    st.stop()  # Stop further processing for non-plant images

                with st.expander("📊 View Detailed Analysis & Recommendations"):
                    st.subheader("Diagnosis Confidence Overview")

                    # -------------------------------
                    # 1. Gauge Chart: Overall Confidence
                    # -------------------------------
                    gauge_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Overall Confidence"},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 50], 'color': "lightcoral"},
                                {'range': [50, 70], 'color': "gold"},
                                {'range': [70, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': confidence
                            }
                        }
                    ))
                    st.plotly_chart(gauge_fig, use_container_width=True)


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
                    🪻 Bell
                </div>
                <div style="background: #f0fdf4;
                            padding: 0.6rem;
                            border-radius: 8px;
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🌲 Cedar
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
                    🍇 Grape
                </div>
                <div style="background: #f0fdf4;
                            padding: 0.6rem;
                            border-radius: 8px;
                            text-align: center;
                            border: 1px solid #dcfce7;">
                    🌽 Maize
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
                    🥔 Potato
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
    <style>
        .hover-text {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }

        .hover-text .tooltip {
            visibility: hidden;
            width: 250px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 5px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8rem;
        }

        .hover-text:hover .tooltip {
            visibility: visible;
            opacity: 1;
        }
    </style>

    <div style="text-align: center; margin-top: 3rem; color: #64748b;">
        <p>🌿 Healthy plants, happier planet</p>
        <p style="font-size: 0.9rem;">
            <span class="hover-text">Upload clear images of plant leaves <span style="color: darkred; font-weight: bold;">(only)</span>
                <span class="tooltip">Users are expected to upload clear images of plant leaves only since the model was not trained on non-plant images.</span>
            </span> for best results
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
