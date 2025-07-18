import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import networkx as nx


class NeuralGraphMemory:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)

    def embed_text(self, text):
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            embeddings = self.clip_model.get_text_features(**inputs)
        return embeddings.cpu().numpy()

    def embed_image_from_url(self, url):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return self.embed_image(image)

    def embed_image(self, image):
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)
        return embeddings.cpu().numpy()

    def add_memory_node(self, node_id, modality, content, embedding):
        self.graph.add_node(node_id, modality=modality, content=content, embedding=embedding)

    def add_edge(self, from_node, to_node, relation="related"):
        self.graph.add_edge(from_node, to_node, relation=relation)

    def retrieve_similar_nodes(self, query_embedding, top_k=3):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        similarities = []
        for node_id, data in self.graph.nodes(data=True):
            if "embedding" in data:
                sim = cosine_similarity(query_embedding, data["embedding"])[0][0]
                similarities.append((node_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def visualize(self):
        import matplotlib.pyplot as plt
        nx.draw(self.graph, with_labels=True, node_color="skyblue", font_weight="bold")
        plt.show()

