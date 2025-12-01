import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Saba Banana Ripeness Detector",
    page_icon="🍌",
    layout="wide"
)

# Custom CSS for mobile responsiveness
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .ripe {
        background: linear-gradient(135deg, #FCD34D 0%, #F59E0B 100%);
        color: #000;
    }
    .not-ripe {
        background: linear-gradient(135deg, #86EFAC 0%, #22C55E 100%);
        color: #000;
    }
    .over-ripe {
        background: linear-gradient(135deg, #FCA5A5 0%, #DC2626 100%);
        color: #fff;
    }
    .model-selector {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        individual_model = tf.keras.models.load_model('Saba_Ripeness_Model_EfficientNetB0_Individual.keras')
        bunch_model = tf.keras.models.load_model('Saba_Ripeness_Model_EfficientNetB0_Bunch.keras')
        return individual_model, bunch_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Preprocess image
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for EfficientNetB0"""
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# Predict ripeness
def predict_ripeness(model, image):
    """Predict ripeness class"""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    
    # Get class index
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    
    # Map to ripeness classes
    classes = ['NOT RIPE', 'RIPE', 'OVER RIPE']
    
    return {
        'class': classes[class_idx],
        'confidence': confidence,
        'probabilities': {
            'NOT RIPE': predictions[0][0] * 100,
            'RIPE': predictions[0][1] * 100,
            'OVER RIPE': predictions[0][2] * 100
        }
    }

# Display prediction
def display_prediction(title, prediction):
    """Display prediction in a styled box"""
    ripeness_class = prediction['class']
    confidence = prediction['confidence']
    
    # Determine style class
    style_class = ripeness_class.lower().replace(' ', '-')
    
    # Emoji mapping
    emoji_map = {
        'NOT RIPE': '🟢',
        'RIPE': '🟡',
        'OVER RIPE': '🔴'
    }
    
    st.markdown(f"""
        <div class="prediction-box {style_class}">
            <h2>{title}</h2>
            <div style="font-size: 4rem;">{emoji_map[ripeness_class]}</div>
            <h1 style="font-size: 3rem; margin: 1rem 0;">{ripeness_class}</h1>
            <h3>Confidence: {confidence:.1f}%</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Show all probabilities
    with st.expander("📊 See detailed probabilities"):
        for cls, prob in prediction['probabilities'].items():
            st.progress(float(prob) / 100.0)
            st.write(f"**{cls}**: {prob:.1f}%")

# Main app
def main():
    st.title("🍌 Saba Banana Ripeness Detector")
    st.markdown("### Choose your detection mode and upload a photo!")
    
    # Load models
    with st.spinner("Loading AI models..."):
        individual_model, bunch_model = load_models()
    
    if individual_model is None or bunch_model is None:
        st.error("⚠️ Models not found! Please ensure both .keras files are in the same directory.")
        st.info("Required files:\n- Saba_Ripeness_Model_EfficientNetB0_Individual.keras\n- Saba_Ripeness_Model_EfficientNetB0_Bunch.keras")
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Model Selection
    st.markdown("---")
    st.markdown('<div class="model-selector">', unsafe_allow_html=True)
    st.markdown("### 🎯 Select Detection Mode")
    
    detection_mode = st.radio(
        "Choose the model based on your banana type:",
        options=["🍌 Single Banana (Individual Model)", 
                 "🍌🍌 Banana Bunch (Bunch Model)", 
                 "🔄 Compare Both Models"],
        help="Individual Model: Best for single bananas\nBunch Model: Best for multiple bananas\nCompare Both: See results from both models side by side"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Image upload options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📷 Take Photo")
        camera_image = st.camera_input("Use camera")
    
    with col2:
        st.markdown("#### 📁 Upload Image")
        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])
    
    # Select which image to use
    image_source = camera_image if camera_image else uploaded_file
    
    if image_source:
        # Display image
        image = Image.open(image_source).convert('RGB')
        st.image(image, caption="Selected Image", use_container_width=True)
        
        # Predict button
        if st.button("🔍 ANALYZE RIPENESS", type="primary"):
            with st.spinner("Analyzing banana ripeness..."):
                st.markdown("---")
                st.markdown("## 📊 Results")
                
                # Individual Model Only
                if detection_mode == "🍌 Single Banana (Individual Model)":
                    individual_pred = predict_ripeness(individual_model, image)
                    display_prediction("🍌 Individual Banana Model", individual_pred)
                    
                    # Recommendation
                    st.markdown("---")
                    st.markdown("### 💡 Recommendation")
                    provide_recommendation(individual_pred['class'])
                
                # Bunch Model Only
                elif detection_mode == "🍌🍌 Banana Bunch (Bunch Model)":
                    bunch_pred = predict_ripeness(bunch_model, image)
                    display_prediction("🍌🍌 Bunch Model", bunch_pred)
                    
                    # Recommendation
                    st.markdown("---")
                    st.markdown("### 💡 Recommendation")
                    provide_recommendation(bunch_pred['class'])
                
                # Compare Both Models
                else:
                    individual_pred = predict_ripeness(individual_model, image)
                    bunch_pred = predict_ripeness(bunch_model, image)
                    
                    # Display predictions side by side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        display_prediction("🍌 Individual Banana Model", individual_pred)
                    
                    with col2:
                        display_prediction("🍌🍌 Bunch Model", bunch_pred)
                    
                    # Model Agreement Analysis
                    st.markdown("---")
                    st.markdown("### 🔍 Model Agreement Analysis")
                    
                    if individual_pred['class'] == bunch_pred['class']:
                        avg_confidence = (individual_pred['confidence'] + bunch_pred['confidence']) / 2
                        st.success(f"✅ **Both models agree!** Prediction: **{individual_pred['class']}** (Average Confidence: {avg_confidence:.1f}%)")
                    else:
                        st.warning(f"⚠️ **Models disagree:**\n- Individual Model: {individual_pred['class']} ({individual_pred['confidence']:.1f}%)\n- Bunch Model: {bunch_pred['class']} ({bunch_pred['confidence']:.1f}%)")
                        st.info("💡 Tip: Use the model that matches your banana type (single vs bunch) for more accurate results.")
                    
                    # Recommendation based on higher confidence
                    st.markdown("---")
                    st.markdown("### 💡 Recommendation")
                    if individual_pred['confidence'] > bunch_pred['confidence']:
                        provide_recommendation(individual_pred['class'])
                    else:
                        provide_recommendation(bunch_pred['class'])
        
        # Reset button
        if st.button("🔄 Analyze Another Banana"):
            st.rerun()
    
    else:
        st.info("👆 Please select a detection mode, then take a photo or upload an image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p><strong>Saba Banana Ripeness Detector</strong></p>
            <p>Powered by EfficientNetB0 Deep Learning Models</p>
            <p>📱 Works on mobile phones | 🌐 Deploy anywhere with Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

def provide_recommendation(ripeness_class):
    """Provide recommendation based on ripeness class"""
    if ripeness_class == 'NOT RIPE':
        st.info("🟢 **NOT RIPE**: Your banana needs more time to ripen. Wait a few more days!")
    elif ripeness_class == 'RIPE':
        st.success("🟡 **RIPE**: Perfect! Your banana is ready to eat. Enjoy!")
    else:
        st.warning("🔴 **OVER RIPE**: Your banana is very ripe. Best for banana bread or smoothies!")

if __name__ == "__main__":
    main()
