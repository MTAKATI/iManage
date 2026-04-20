import os
import shutil
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from elasticsearch import Elasticsearch
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Import the sync logic
from manage_data import sync_dataset

app = FastAPI()
es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "image_index"
DATASET_PATH = "dataset/similarity_dataset"

class FeatureExtractor:
    def __init__(self):
        # Explicit input_shape avoids the UserWarning you saw earlier
        base_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling='avg')
        self.model = Model(inputs=base_model.input, outputs=base_model.output)

    def extract(self, img_path):
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array[np.newaxis, ...])
            features = self.model.predict(img_array, verbose=0)
            features = features.flatten()
            features = features / np.linalg.norm(features)
            return features.tolist()
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return None

fe = FeatureExtractor()

@app.on_event("startup")
async def startup_event():
    # Automatically syncs your images every time the server starts
    sync_dataset(fe, es, INDEX_NAME, DATASET_PATH)

def search_similar_images(query_vector, top_k=5):
    knn_query = {
        "field": "feature_vector",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": 100
    }
    try:
        res = es.search(index=INDEX_NAME, body={"knn":knn_query})
        return [
            {"image": hit["_source"].get("image_path", "unknown"), "score": hit["_score"]} 
            for hit in res["hits"]["hits"]
        ]
    except Exception as e:
        print(f"Search Error: {e}")
        return []

@app.post("/search-image")
async def search_image(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", file.filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        vector = fe.extract(temp_path)
        results = search_similar_images(vector)
        return {"results": results}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)