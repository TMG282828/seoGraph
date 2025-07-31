"""
Basic structure test to verify test framework setup.

Tests basic imports and structure without triggering full module initialization.
"""

import pytest
import os
import sys

def test_test_directory_structure():
    """Test that test directory structure is correct."""
    current_dir = os.path.dirname(__file__)
    
    # Check that we're in the right place
    assert "test_content_generation" in current_dir
    
    # Check that test files exist
    test_files = [
        "test_agent.py",
        "test_tools.py", 
        "test_rag_tools.py",
        "test_prompts.py"
    ]
    
    for test_file in test_files:
        file_path = os.path.join(current_dir, test_file)
        assert os.path.exists(file_path), f"Test file {test_file} should exist"

def test_project_structure():
    """Test that project structure is accessible."""
    # Get project root (3 levels up from test file)
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    
    # Check key directories exist
    src_dir = os.path.join(project_root, "src")
    agents_dir = os.path.join(src_dir, "agents")
    content_gen_dir = os.path.join(agents_dir, "content_generation")
    
    assert os.path.exists(src_dir), "src directory should exist"
    assert os.path.exists(agents_dir), "agents directory should exist"  
    assert os.path.exists(content_gen_dir), "content_generation directory should exist"
    
    # Check key files exist
    agent_file = os.path.join(content_gen_dir, "agent.py")
    tools_file = os.path.join(content_gen_dir, "tools.py")
    rag_tools_file = os.path.join(content_gen_dir, "rag_tools.py")
    prompts_file = os.path.join(content_gen_dir, "prompts.py")
    
    assert os.path.exists(agent_file), "agent.py should exist"
    assert os.path.exists(tools_file), "tools.py should exist"
    assert os.path.exists(rag_tools_file), "rag_tools.py should exist"
    assert os.path.exists(prompts_file), "prompts.py should exist"

def test_module_structure_without_import():
    """Test module structure by checking file contents without importing."""
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    
    # Check agent.py contains expected classes
    agent_file = os.path.join(project_root, "src/agents/content_generation/agent.py")
    with open(agent_file, 'r') as f:
        agent_content = f.read()
        
    assert "class ContentGenerationAgent" in agent_content
    assert "class ContentGenerationRequest" in agent_content
    
    # Check tools.py contains expected class
    tools_file = os.path.join(project_root, "src/agents/content_generation/tools.py")
    with open(tools_file, 'r') as f:
        tools_content = f.read()
        
    assert "class ContentGenerationTools" in tools_content
    
    # Check rag_tools.py contains expected class
    rag_tools_file = os.path.join(project_root, "src/agents/content_generation/rag_tools.py")
    with open(rag_tools_file, 'r') as f:
        rag_tools_content = f.read()
        
    assert "class RAGTools" in rag_tools_content
    
    # Check prompts.py contains expected class
    prompts_file = os.path.join(project_root, "src/agents/content_generation/prompts.py")  
    with open(prompts_file, 'r') as f:
        prompts_content = f.read()
        
    assert "class ContentGenerationPrompts" in prompts_content

def test_python_path_setup():
    """Test that Python path is set up correctly for imports."""
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    
    # Add project root to Python path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Test that we can find the modules (without importing them fully)
    import importlib.util
    
    # Check that we can locate the module files
    agent_spec = importlib.util.find_spec("src.agents.content_generation.agent")
    assert agent_spec is not None, "Should be able to find agent module"
    
    tools_spec = importlib.util.find_spec("src.agents.content_generation.tools") 
    assert tools_spec is not None, "Should be able to find tools module"

if __name__ == "__main__":
    pytest.main([__file__])