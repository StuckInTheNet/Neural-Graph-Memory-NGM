#!/usr/bin/env python3
"""
Test script to verify all NGM fixes are working
"""

import sys
import os
sys.path.append('SRC')

from memory import NeuralGraphMemory
from load_data import load_graph_data

def test_basic_functionality():
    """Test basic NGM functionality"""
    print("🧪 Testing basic NGM functionality...")
    
    # Initialize NGM
    ngm = NeuralGraphMemory()
    print("✅ NGM initialized successfully")
    
    # Test text embedding
    text_emb = ngm.embed_text("This is a test")
    print(f"✅ Text embedding shape: {text_emb.shape}")
    
    # Test adding a single node
    ngm.add_memory_node("test_node", "text", "Test content", text_emb)
    print("✅ Single node added successfully")
    
    # Test retrieval
    results = ngm.retrieve_similar_nodes(text_emb, top_k=1)
    print(f"✅ Retrieval working, found {len(results)} results")
    
    # Test query method
    answer = ngm.query("test")
    print(f"✅ Query method working: '{answer}'")
    
    return True

def test_data_loading():
    """Test data loading functionality"""
    print("\n🧪 Testing data loading...")
    
    # Load data
    data = load_graph_data('data')
    print(f"✅ Data loaded: {len(data['nodes'])} nodes, {len(data['edges'])} edges")
    
    # Initialize NGM and load data
    ngm = NeuralGraphMemory()
    ngm.add_nodes(data['nodes'])
    ngm.add_edges(data['edges'])
    print(f"✅ Graph constructed with {ngm.graph.number_of_nodes()} nodes and {ngm.graph.number_of_edges()} edges")
    
    return True

def test_multimodal_fusion():
    """Test multimodal fusion functionality"""
    print("\n🧪 Testing multimodal fusion...")
    
    ngm = NeuralGraphMemory()
    
    # Test different modality enhancements
    text_emb = ngm.embed_text("Sample text")
    image_enhanced = ngm._enhance_embedding_for_modality(text_emb, 'image')
    audio_enhanced = ngm._enhance_embedding_for_modality(text_emb, 'audio')
    
    print("✅ Modality-specific enhancements working")
    
    # Test multimodal fusion
    fused = ngm.fuse_multimodal_embeddings(text_emb, image_enhanced, audio_enhanced)
    print(f"✅ Multimodal fusion working, output shape: {fused.shape}")
    
    return True

def test_temporal_weighting():
    """Test temporal weighting functionality"""
    print("\n🧪 Testing temporal weighting...")
    
    ngm = NeuralGraphMemory()
    
    # Add two nodes with timestamps
    emb = ngm.embed_text("test")
    ngm.add_memory_node("node1", "text", "First node", emb, "2025-01-01T10:00:00")
    ngm.add_memory_node("node2", "text", "Second node", emb, "2025-01-01T11:00:00")
    
    # Test temporal weight calculation
    weight = ngm._calculate_temporal_weight("node1", "node2")
    print(f"✅ Temporal weighting working, weight: {weight:.4f}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting NGM functionality tests...\n")
    
    try:
        test_basic_functionality()
        test_data_loading()
        test_multimodal_fusion()
        test_temporal_weighting()
        
        print("\n🎉 All tests passed! NGM is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)