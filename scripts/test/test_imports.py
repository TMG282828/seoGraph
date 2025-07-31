#!/usr/bin/env python3
"""
Test imports for content generation modules.
"""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    try:
        print("Testing content generation agent import...")
        from src.agents.content_generation.agent import ContentGenerationAgent, ContentGenerationRequest
        print("✓ Agent imports successful")
        
        print("Testing tools import...")
        from src.agents.content_generation.tools import ContentGenerationTools
        print("✓ Tools imports successful")
        
        print("Testing RAG tools import...")
        from src.agents.content_generation.rag_tools import RAGTools
        print("✓ RAG tools imports successful")
        
        print("Testing prompts import...")
        from src.agents.content_generation.prompts import ContentGenerationPrompts
        print("✓ Prompts imports successful")
        
        print("Testing base agent import...")
        from src.agents.base_agent import AgentContext, AgentResult
        print("✓ Base agent imports successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✓ All imports successful - comprehensive tests should work")
    else:
        print("\n✗ Import issues detected")
    sys.exit(0 if success else 1)