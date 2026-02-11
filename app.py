import streamlit as st
import torch
import torchvision.transforms as tfs
from PIL import Image
from model import FFA 
import os
import time

# --- Page Config ---
st.set_page_config(
    page_title="Fog Removal System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
MODEL_PATH = 'ffa_epoch_10.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Load Model (Cached) ---
@st.cache_resource
def load_model():
    try:
        # 1. Initialize Structure
        net = FFA(gps=3, blocks=19)
        net.to(DEVICE)
        
        # 2. Load Weights
        if os.path.exists(MODEL_PATH):
            # map_location ensures it works on both your PC and Friend's Laptop
            net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            net.eval()
            return net
        else:
            st.error(f"Model file '{MODEL_PATH}' not found! Please check the folder.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

net = load_model()

# --- Sidebar ---
st.sidebar.title("System Status")
st.sidebar.success(f"Running on: **{DEVICE.upper()}**")
st.sidebar.info("FFA-Net (Feature Fusion Attention Network)")

# --- Main Interface ---
st.title("🚗 Autonomous Visibility Enhancement")
st.markdown("### Deep Learning Fog Removal")

col1, col2 = st.columns(2)

# 1. Upload
uploaded_file = st.sidebar.file_uploader("Upload a Foggy Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read Image
    original_image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Original Input")
        st.image(original_image, use_container_width=True)

    # 2. Dehaze Button
    if st.sidebar.button("✨ Dehaze Image", type="primary"):
        if net is None:
            st.error("Model not loaded.")
        else:
            with st.spinner('Restoring visibility...'):
                start_time = time.time()
                output_img = None
                
                try:
                    # --- Preprocessing ---
                    # Resize to multiple of 16 to prevent dimension errors
                    w, h = original_image.size
                    w_new = w - (w % 16)
                    h_new = h - (h % 16)
                    
                    img_resized = original_image.resize((w_new, h_new))
                    transform = tfs.Compose([tfs.ToTensor()])
                    img_tensor = transform(img_resized).unsqueeze(0).to(DEVICE)
                    
                    # --- Inference ---
                    with torch.no_grad():
                        output_tensor = net(img_tensor)
                    
                    # --- Postprocessing ---
                    output_tensor = output_tensor.squeeze(0).cpu()
                    # Clamp to fix any "white pixel" artifacts
                    output_tensor = torch.clamp(output_tensor, 0, 1)
                    output_img = tfs.ToPILImage()(output_tensor)
                    
                    end_time = time.time()

                except RuntimeError as e:
                    st.error(f"⚠️ Model Error: {e}")
                    st.warning("This is likely the 'Shape Mismatch' error. Please fix model.py!")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                
            # 3. Display Result (Only if successful)
            if output_img is not None:
                with col2:
                    st.subheader("Dehazed Output")
                    st.image(output_img, use_container_width=True)
                
                st.success(f"Processing complete in {end_time - start_time:.3f} seconds!")

else:
    # Placeholder when no image is uploaded
    with col1:
        st.info("Upload an image from the sidebar to begin.")