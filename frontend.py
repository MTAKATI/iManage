import streamlit as st
import requests
from PIL import Image
import os

st.set_page_config(page_title="iManage Search", layout="wide")

st.title("🖼️ iManage: Visual Similarity Search")
st.write("Upload an image to find similar items in the dataset.")

# 1. Sidebar - Upload Section
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

# 2. Main Area - Results Section
if uploaded_file is not None:
    # Display the query image
    st.subheader("Query Image")
    st.image(uploaded_file, width=300)

    # Send to FastAPI
    if st.button("Search Similar Images"):
        with st.spinner("Searching..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post("http://127.0.0.1:5000/search-image", files=files)

            if response.status_code == 200:
                results = response.json().get("results", [])
                
                if not results:
                    st.warning("No similar images found.")
                else:
                    st.subheader("Top Matches")
                    # Create a grid for results
                    cols = st.columns(3)
                    for idx, res in enumerate(results):
                        # res['image'] usually looks like 'dataset/similarity_dataset\\cat.jpg'
                        img_path = res['image']
                        score = res['score']
                        
                        # Handle path formatting for Windows/Linux
                        clean_path = img_path.replace("\\", "/")
                        
                        with cols[idx % 3]:
                            if os.path.exists(clean_path):
                                st.image(clean_path, use_container_width=True, caption=f"Match Score: {score:.4f}")
                                st.write(f"Path: {os.path.basename(clean_path)}")
                            else:
                                st.error(f"Image not found: {os.path.basename(clean_path)}")
            else:
                st.error("Error connecting to the backend API.")