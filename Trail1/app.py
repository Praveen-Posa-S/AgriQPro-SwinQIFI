import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import sys

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import AgriQPro

# Constants
CLASSES = ['Bacterial Leaf Disease', 'Dried Leaf', 'Fungal Brown Spot Disease', 'Healthy_Leaf', 'Leaf_Rot', 'Leaf_Spot']
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Page config
st.set_page_config(page_title="AgriQPro - Plant Disease Detection", page_icon="🌿")

st.title("🌿 AgriQPro – Quantum Driven Precision Optimization for Smart Agriculture")
st.write("Upload a Betel Vine leaf image to detect diseases using the SwinV2 + QIFI model.")

@st.cache_resource
def load_model(checkpoint_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model with correct parameters (img_size=224, window_size=7)
    model = AgriQPro(num_classes=num_classes, backbone_name='swinv2_tiny_window8_256')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        st.error(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return None
    
    model.to(device)
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    return transform(image).unsqueeze(0)

# Sidebar
st.sidebar.header("Model Settings")
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
checkpoint_path = st.sidebar.text_input("Checkpoint Path", os.path.join("checkpoints", "best_model.pth"))

# Load Model
model = load_model(checkpoint_path, len(CLASSES))

# Input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        if st.button("Predict Disease"):
            with st.spinner("Classifying..."):
                # Preprocess
                input_tensor = preprocess_image(image)
                device = next(model.parameters()).device
                input_tensor = input_tensor.to(device)
                
                # Predict
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probs, 1)
                
                # Result
                class_name = CLASSES[predicted_class.item()]
                conf_score = confidence.item() * 100
                
                st.success(f"**Prediction:** {class_name}")
                st.info(f"**Confidence:** {conf_score:.2f}%")
                
                # Detailed probabilities Graph
                st.write("---")
                st.write("**Confidence Scores:**")
                probs_np = probs.cpu().numpy()[0]
                
                # Create DataFrame for plotting
                df = pd.DataFrame({
                    'Disease': CLASSES,
                    'Confidence': probs_np
                })
                
                st.bar_chart(df.set_index('Disease'))
