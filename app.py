import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd

# Page config
st.set_page_config(
    page_title="Pet Classifier 🐾",
    page_icon="🐾",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #FF6B6B;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #4ECDC4;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# Pet mapping from ImageNet classes
PET_CLASSES = {
    'dog': ['Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu', 
            'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 
            'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'Labrador_retriever',
            'golden_retriever', 'bulldog', 'pug', 'dalmatian', 'corgi', 'husky',
            'German_shepherd', 'collie', 'poodle', 'boxer', 'Great_Dane'],
    'cat': ['tabby', 'tiger_cat', 'Persian_cat', 'Siamese_cat', 'Egyptian_cat'],
    'bird': ['robin', 'jay', 'bald_eagle', 'vulture', 'great_grey_owl', 'cock',
             'hen', 'ostrich', 'brambling', 'goldfinch', 'house_finch', 'junco',
             'indigo_bunting', 'magpie', 'chickadee', 'water_ouzel', 'kite',
             'hummingbird', 'peacock', 'toucan', 'hornbill', 'lorikeet'],
    'rabbit': ['hare', 'wood_rabbit'],
    'hamster': ['hamster'],
    'guinea_pig': ['guinea_pig']
}

# Emoji mapping
PET_EMOJI = {
    'dog': '🐶',
    'cat': '🐱',
    'bird': '🐦',
    'rabbit': '🐰',
    'hamster': '🐹',
    'guinea_pig': '🐹'
}

@st.cache_resource
def load_model():
    """Load pretrained MobileNetV2 model"""
    model = MobileNetV2(weights='imagenet', include_top=True)
    return model

def preprocess_image(img):
    """Preprocess image for model prediction"""
    # Resize image to 224x224
    img = img.resize((224, 224))
    # Convert to array
    img_array = image.img_to_array(img)
    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess for MobileNetV2
    img_array = preprocess_input(img_array)
    return img_array

def classify_pet(model, img):
    """Classify the pet image"""
    # Preprocess
    processed_img = preprocess_image(img)
    
    # Predict
    predictions = model.predict(processed_img, verbose=0)
    
    # Decode predictions
    decoded = decode_predictions(predictions, top=10)[0]
    
    # Map to pet categories
    pet_scores = {pet: 0.0 for pet in PET_CLASSES.keys()}
    
    for _, class_name, score in decoded:
        for pet_type, class_list in PET_CLASSES.items():
            if class_name in class_list:
                pet_scores[pet_type] += score
    
    # Get top prediction
    if max(pet_scores.values()) > 0:
        top_pet = max(pet_scores, key=pet_scores.get)
        confidence = pet_scores[top_pet] * 100
        return top_pet, confidence, pet_scores
    else:
        return None, 0, pet_scores

# Header
st.markdown('<p class="main-header">🐾 Pet Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a pet image and let AI identify it!</p>', unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses a **MobileNetV2** CNN model 
    pretrained on ImageNet to classify pet images.
    
    **Supported pets:**
    - 🐶 Dogs
    - 🐱 Cats
    - 🐦 Birds
    - 🐰 Rabbits
    - 🐹 Hamsters & Guinea Pigs
    
    **Tech Stack:**
    - TensorFlow/Keras
    - Streamlit
    - MobileNetV2 (Transfer Learning)
    """)
    
    st.header("📊 Model Info")
    st.write("""
    - **Architecture:** MobileNetV2
    - **Parameters:** ~3.5M
    - **Training Data:** ImageNet
    - **Input Size:** 224x224
    """)

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader(
        "Choose a pet image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of your pet"
    )

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert('RGB')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption='Uploaded Image', use_container_width=True)
    
    # Load model
    with st.spinner('🔄 Loading AI model...'):
        model = load_model()
    
    # Classify
    with st.spinner('🤔 Analyzing your pet...'):
        pet_type, confidence, all_scores = classify_pet(model, img)
    
    # Display results
    st.markdown("---")
    
    if pet_type and confidence > 5:
        # Main prediction
        st.markdown("### 🎯 Prediction Result")
        
        emoji = PET_EMOJI.get(pet_type, '🐾')
        
        result_col1, result_col2 = st.columns([1, 2])
        with result_col1:
            st.markdown(f"<h1 style='text-align: center; font-size: 5em;'>{emoji}</h1>", 
                       unsafe_allow_html=True)
        with result_col2:
            st.markdown(f"<h2 style='color: #FF6B6B;'>{pet_type.upper()}</h2>", 
                       unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #4ECDC4;'>Confidence: {confidence:.1f}%</h3>", 
                       unsafe_allow_html=True)
        
        # Confidence bar
        st.progress(float(confidence / 100))
        
        # All scores
        st.markdown("### 📊 All Predictions")
        
        # Create DataFrame for visualization
        df_scores = pd.DataFrame({
            'Pet Type': [f"{PET_EMOJI.get(k, '🐾')} {k.capitalize()}" for k, v in all_scores.items() if v > 0],
            'Confidence (%)': [v * 100 for v in all_scores.values() if v > 0]
        }).sort_values('Confidence (%)', ascending=False)
        
        if not df_scores.empty:
            st.dataframe(
                df_scores.style.background_gradient(cmap='RdYlGn', subset=['Confidence (%)']),
                hide_index=True,
                use_container_width=True
            )
        
        # Confidence interpretation
        if confidence >= 80:
            st.success("✅ High confidence prediction!")
        elif confidence >= 50:
            st.info("ℹ️ Moderate confidence. The model is fairly certain.")
        else:
            st.warning("⚠️ Low confidence. Try uploading a clearer image.")
    else:
        st.error("❌ Could not identify a pet in this image. Please upload an image of a dog, cat, bird, rabbit, hamster, or guinea pig.")

else:
    # Show example
    st.info("👆 Upload an image above to get started!")
    
    st.markdown("### 💡 Tips for best results:")
    st.markdown("""
    - Use clear, well-lit images
    - Make sure the pet is the main subject
    - Avoid blurry or very small images
    - Front-facing or side profile works best
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>Built with ❤️ using TensorFlow & Streamlit</p>",
    unsafe_allow_html=True
)