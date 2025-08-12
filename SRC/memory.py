import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import networkx as nx
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


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

    def add_memory_node(self, node_id, modality, content, embedding, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        self.graph.add_node(node_id, modality=modality, content=content, embedding=embedding, timestamp=timestamp)

    def add_edge(self, from_node, to_node, relation="related", weight=1.0):
        self.graph.add_edge(from_node, to_node, relation=relation, weight=weight)
    
    def add_nodes(self, nodes):
        """Add multiple nodes from data structure"""
        for node in nodes:
            # Extract multimodal embedding based on node type
            embedding = self._create_node_embedding(node)
            self.add_memory_node(
                node_id=node['id'],
                modality=node.get('type', 'text'),
                content=node.get('description', ''),
                embedding=embedding,
                timestamp=node.get('timestamp')
            )
    
    def add_edges(self, edges):
        """Add multiple edges from data structure"""
        for edge in edges:
            # Calculate temporal weight if both nodes exist
            weight = self._calculate_temporal_weight(edge['source'], edge['target'])
            self.add_edge(
                from_node=edge['source'],
                to_node=edge['target'],
                relation=edge.get('relation', 'temporal'),
                weight=weight
            )
    
    def _create_node_embedding(self, node):
        """Create embedding based on node description and type"""
        description = node.get('description', '')
        node_type = node.get('type', 'text')
        
        if description:
            text_embedding = self.embed_text(description)
            
            # Enhanced multimodal fusion based on node type
            if node_type == 'image':
                # For image nodes, we'd ideally have the actual image
                # For now, use text description but could be extended
                return self._enhance_embedding_for_modality(text_embedding, 'image')
            elif node_type == 'audio':
                return self._enhance_embedding_for_modality(text_embedding, 'audio')
            elif node_type == 'event':
                return self._enhance_embedding_for_modality(text_embedding, 'event')
            else:
                return text_embedding
        else:
            # Return zero embedding if no description
            return np.zeros((1, 512))
    
    def _enhance_embedding_for_modality(self, base_embedding, modality):
        """Apply modality-specific enhancement to embeddings"""
        # Simple modality-specific linear transformation
        # In a full implementation, this would be learned parameters
        modality_weights = {
            'image': 1.2,
            'audio': 1.1,
            'event': 1.15,
            'text': 1.0
        }
        
        weight = modality_weights.get(modality, 1.0)
        return base_embedding * weight
    
    def fuse_multimodal_embeddings(self, text_embedding, image_embedding=None, audio_embedding=None, metadata=None):
        """
        Multimodal fusion function as described in the paper
        Implements a simplified version of the gated cross-modal transformer
        """
        embeddings = [text_embedding]
        weights = [1.0]
        
        if image_embedding is not None:
            embeddings.append(image_embedding)
            weights.append(1.2)  # Slight boost for visual information
            
        if audio_embedding is not None:
            embeddings.append(audio_embedding)
            weights.append(1.1)  # Slight boost for audio information
        
        # Weighted fusion (simplified version of attention-based integration)
        if len(embeddings) == 1:
            return embeddings[0]
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted average with learned attention (simplified)
        fused = np.zeros_like(embeddings[0])
        for emb, weight in zip(embeddings, weights):
            fused += emb * weight
        
        return fused
    
    def _calculate_temporal_weight(self, source_id, target_id):
        """Calculate temporal weight between two nodes"""
        if source_id in self.graph.nodes and target_id in self.graph.nodes:
            source_time = self.graph.nodes[source_id].get('timestamp')
            target_time = self.graph.nodes[target_id].get('timestamp')
            
            if source_time and target_time:
                try:
                    source_dt = datetime.fromisoformat(source_time.replace('Z', '+00:00'))
                    target_dt = datetime.fromisoformat(target_time.replace('Z', '+00:00'))
                    time_diff = abs((target_dt - source_dt).total_seconds())
                    # Exponential decay: closer in time = higher weight
                    return np.exp(-time_diff / 3600)  # 1-hour decay constant
                except:
                    pass
        return 1.0
    
    def query(self, query_text, top_k=3, use_traversal=True):
        """Query the memory graph and return relevant information"""
        query_embedding = self.embed_text(query_text)
        
        if use_traversal:
            return self._graph_traversal_retrieval(query_embedding, top_k)
        else:
            return self._similarity_retrieval(query_embedding, top_k)
    
    def _similarity_retrieval(self, query_embedding, top_k):
        """Basic similarity-based retrieval"""
        similarities = self.retrieve_similar_nodes(query_embedding, top_k)
        
        if similarities:
            # Return the content of the most similar node
            best_node_id = similarities[0][0]
            if best_node_id in self.graph.nodes:
                return self.graph.nodes[best_node_id].get('content', 'No content found')
        
        return "No relevant memory found"
    
    def _graph_traversal_retrieval(self, query_embedding, top_k):
        """Graph traversal-based retrieval with attention weighting"""
        # First get initial candidates
        initial_candidates = self.retrieve_similar_nodes(query_embedding, top_k * 2)
        
        if not initial_candidates:
            return "No relevant memory found"
        
        # Expand search through graph connections
        expanded_nodes = set()
        for node_id, _ in initial_candidates:
            expanded_nodes.add(node_id)
            # Add neighbors
            for neighbor in self.graph.neighbors(node_id):
                expanded_nodes.add(neighbor)
        
        # Re-rank expanded set
        final_similarities = []
        for node_id in expanded_nodes:
            if node_id in self.graph.nodes and 'embedding' in self.graph.nodes[node_id]:
                node_embedding = self.graph.nodes[node_id]['embedding']
                sim = cosine_similarity(query_embedding, node_embedding)[0][0]
                
                # Apply graph-based boosting
                degree_boost = 1 + (self.graph.degree(node_id) * 0.1)
                final_similarities.append((node_id, sim * degree_boost))
        
        if final_similarities:
            final_similarities.sort(key=lambda x: x[1], reverse=True)
            best_node_id = final_similarities[0][0]
            
            # Return rich context including connected nodes
            main_content = self.graph.nodes[best_node_id].get('content', '')
            connected_content = []
            
            for neighbor in self.graph.neighbors(best_node_id):
                if 'content' in self.graph.nodes[neighbor]:
                    connected_content.append(self.graph.nodes[neighbor]['content'])
            
            if connected_content:
                return f"{main_content}. Related: {'; '.join(connected_content[:2])}"
            else:
                return main_content
        
        return "No relevant memory found"

    def retrieve_similar_nodes(self, query_embedding, top_k=3):
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

