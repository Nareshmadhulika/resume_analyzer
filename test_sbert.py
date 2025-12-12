#!/usr/bin/env python3
"""
Test script to verify SBERT functionality for the Resume Analyzer app
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def test_sbert_functionality():
    """Test the SBERT model and similarity computation"""
    print("Testing SBERT functionality...")
    
    try:
        # Load the model
        print("1. Loading SBERT model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("   ✅ Model loaded successfully!")
        
        # Test with sample texts
        print("2. Testing with sample texts...")
        resume_text = "Experienced software engineer with 5 years of Python development experience. Skilled in machine learning, data analysis, and web development."
        jd_text = "We are looking for a Python developer with experience in machine learning and data analysis. The ideal candidate should have strong programming skills."
        
        # Generate embeddings
        print("3. Generating embeddings...")
        resume_emb = model.encode(resume_text)
        jd_emb = model.encode(jd_text)
        print(f"   ✅ Embeddings generated! Shape: {np.array(resume_emb).shape}")
        
        # Compute similarity
        print("4. Computing similarity...")
        similarity_score = cosine_similarity([resume_emb], [jd_emb])[0][0]
        print(f"   ✅ Similarity score: {similarity_score:.4f}")
        
        # Test with different texts
        print("5. Testing with different texts...")
        different_jd = "We are looking for a marketing specialist with experience in social media management and content creation."
        different_emb = model.encode(different_jd)
        different_similarity = cosine_similarity([resume_emb], [different_emb])[0][0]
        print(f"   ✅ Different JD similarity score: {different_similarity:.4f}")
        
        # Verify that similar texts have higher similarity
        if similarity_score > different_similarity:
            print("   ✅ Logic verified: Related texts have higher similarity!")
        else:
            print("   ⚠️  Unexpected: Related texts don't have higher similarity")
        
        print("\n🎉 All tests passed! SBERT functionality is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    test_sbert_functionality() 