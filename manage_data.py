import os

def sync_dataset(fe, es, index_name, dataset_path):
    """
    Checks the local folder and syncs new images to Elasticsearch 
    without recreating the whole index every time.
    """
    # 1. Ensure the index and mapping exist
    if not es.indices.exists(index=index_name):
        print(f"Creating fresh index: {index_name}")
        mapping = {
            "mappings": {
                "properties": {
                    "image_path": {"type": "keyword"},
                    "feature_vector": {
                        "type": "dense_vector",
                        "dims": 1280,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        es.indices.create(index=index_name, body=mapping)

    # 2. Sync images
    print(f"Checking for new images in {dataset_path}...")
    indexed_count = 0
    
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dataset_path, filename)
            
            # Use a term query for exact path matching
            search_check = es.search(
                index=index_name, 
                body={"query": {"term": {"image_path": img_path}}}
            )
            
            if search_check['hits']['total']['value'] == 0:
                print(f"➕ Indexing new image: {filename}")
                vector = fe.extract(img_path)
                
                # Convert numpy array to list if necessary
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()
                
                es.index(index=index_name, document={
                    "image_path": img_path,
                    "feature_vector": vector
                })
                indexed_count += 1
    
    print(f"🏁 Sync complete. Added {indexed_count} new images.")