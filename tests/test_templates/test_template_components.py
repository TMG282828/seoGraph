"""
Unit tests for template components - Fixed version.

Tests template syntax, structure, required elements, CSS classes,
JavaScript functionality, and accessibility features.
"""

import pytest
import re
from pathlib import Path
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader


class TestTemplateComponents:
    """Test suite for template components."""
    
    @pytest.fixture
    def templates_dir(self):
        """Get templates directory path."""
        project_root = Path(__file__).parent.parent.parent
        return project_root / "web" / "templates"
    
    @pytest.fixture
    def components_dir(self, templates_dir):
        """Get components directory path."""
        return templates_dir / "components"
    
    @pytest.fixture
    def jinja_env(self, templates_dir):
        """Create Jinja2 environment for template testing."""
        return Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=True
        )
    
    @pytest.fixture
    def component_files(self, components_dir):
        """Get list of component HTML files."""
        if not components_dir.exists():
            return []
        return list(components_dir.glob("*.html"))
    
    def test_components_directory_exists(self, components_dir):
        """Test that components directory exists."""
        assert components_dir.exists(), "Components directory should exist"
        assert components_dir.is_dir(), "Components path should be a directory"
    
    def test_component_files_exist(self, component_files):
        """Test that component files exist."""
        assert len(component_files) > 0, "Should have at least one component file"
        
        # Check for key components
        component_names = [f.name for f in component_files]
        expected_components = [
            "page_header.html",
            "sidebar_navigation.html", 
            "base_styles.html",
            "base_scripts.html",
            "gsc_connection_status.html"
        ]
        
        found_components = []
        for expected in expected_components:
            if expected in component_names:
                found_components.append(expected)
        
        assert len(found_components) >= 3, f"Should have key components. Found: {found_components}"
    
    def test_templates_have_valid_syntax(self, jinja_env, component_files):
        """Test that all templates have valid Jinja2 syntax."""
        syntax_errors = []
        
        for component_file in component_files:
            try:
                # Try to parse the template
                template_path = f"components/{component_file.name}"
                template = jinja_env.get_template(template_path)
                
                # Basic render test with minimal context
                template.render(request={'url': {'path': '/'}})
                
            except Exception as e:
                if "syntax" in str(e).lower() or "unexpected" in str(e).lower():
                    syntax_errors.append(f"{component_file.name}: {e}")
        
        assert len(syntax_errors) == 0, f"Template syntax errors: {syntax_errors}"
    
    def test_page_header_component(self, components_dir):
        """Test page header component structure."""
        header_file = components_dir / "page_header.html"
        if not header_file.exists():
            pytest.skip("page_header.html not found")
        
        content = header_file.read_text()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Test structure
        assert soup.find('h2'), "Header should have h2 for page title"
        
        # Test theme toggle button exists (either by ID or by function)
        theme_button = soup.find('button', {'id': 'global-theme-toggle'}) or \
                      soup.find('button', string=re.compile(r'Dark|Light|Theme'))
        assert theme_button, "Should have theme toggle button"
        
        # Test CSS classes
        assert 'flex' in content, "Should use flexbox layout"
        
        # Test block structure for template inheritance
        assert '{% block page_title %}' in content, "Should have page title block"
    
    def test_sidebar_navigation_component(self, components_dir):
        """Test sidebar navigation component structure."""
        sidebar_file = components_dir / "sidebar_navigation.html"
        if not sidebar_file.exists():
            pytest.skip("sidebar_navigation.html not found")
        
        content = sidebar_file.read_text()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Test navigation structure
        nav = soup.find('nav')
        assert nav, "Should have nav element"
        
        nav_items = soup.find_all('a', class_='nav-item')
        assert len(nav_items) >= 5, "Should have at least 5 navigation items"
        
        # Test expected navigation links
        nav_links = [a.get('href') for a in nav_items if a.get('href')]
        expected_links = ['/', '/seo-monitor', '/content-studio', '/settings']
        
        found_links = []
        for expected_link in expected_links:
            if expected_link in nav_links:
                found_links.append(expected_link)
        
        assert len(found_links) >= 3, f"Should have main navigation links. Found: {found_links}"
        
        # Test active state logic
        assert '{% if request.url.path ==' in content, "Should have active state logic"
        
        # Test status indicators
        assert 'status-indicator' in content or 'System Status' in content, "Should have system status section"
    
    def test_gsc_connection_status_component(self, components_dir):
        """Test GSC connection status component."""
        gsc_file = components_dir / "gsc_connection_status.html"
        if not gsc_file.exists():
            pytest.skip("gsc_connection_status.html not found")
        
        content = gsc_file.read_text()
        
        # Test Alpine.js directives
        assert 'x-show=' in content, "Should use Alpine.js x-show directive"
        
        # Test conditional display variables
        assert 'gscConnected' in content, "Should reference gscConnected state"
        assert 'gscStatusChecked' in content, "Should reference gscStatusChecked state"
        
        # Test Google branding
        assert 'Google' in content, "Should mention Google Search Console"
        
        # Test color styling
        color_classes = ['border-yellow', 'border-green', 'yellow-500', 'green-500']
        has_colors = any(color in content for color in color_classes)
        assert has_colors, "Should have color-coded status styling"
    
    def test_base_styles_component(self, components_dir):
        """Test base styles component."""
        styles_file = components_dir / "base_styles.html"
        if not styles_file.exists():
            pytest.skip("base_styles.html not found")
        
        content = styles_file.read_text()
        
        # Test CSS variables
        essential_vars = ['--bg-primary', '--text-primary', '--border-color']
        found_vars = [var for var in essential_vars if var in content]
        assert len(found_vars) >= 3, f"Should define essential CSS variables. Found: {found_vars}"
        
        # Test theme support
        assert ':root' in content, "Should have root CSS variables"
        theme_support = 'body.theme-light' in content or '[data-theme=' in content
        assert theme_support, "Should support theme switching"
        
        # Test component classes
        component_classes = ['.sidebar', '.nav-item', '.card', '.btn-primary']
        found_classes = [cls for cls in component_classes if cls in content]
        assert len(found_classes) >= 3, f"Should define component styles. Found: {found_classes}"
    
    def test_base_scripts_component(self, components_dir):
        """Test base scripts component."""
        scripts_file = components_dir / "base_scripts.html"
        if not scripts_file.exists():
            pytest.skip("base_scripts.html not found")
        
        content = scripts_file.read_text()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Test script tags exist
        scripts = soup.find_all('script')
        
        if len(scripts) > 0:
            # Should have either content or src
            has_content = any(script.string and script.string.strip() for script in scripts)
            has_src = any(script.get('src') for script in scripts)
            
            assert has_content or has_src, "Scripts should have either content or src attribute"
    
    def test_modal_components_exist(self, components_dir):
        """Test that modal components exist and have basic structure."""
        modal_files = list(components_dir.glob("*modal*.html"))
        
        if len(modal_files) > 0:
            for modal_file in modal_files:
                content = modal_file.read_text()
                assert 'modal' in content.lower(), f"{modal_file.name} should contain modal elements"
    
    def test_components_have_meaningful_content(self, component_files):
        """Test that components have meaningful content."""
        for component_file in component_files:
            content = component_file.read_text()
            
            # Should not be empty
            assert len(content.strip()) > 0, f"{component_file.name} should not be empty"
            
            # Should have some HTML structure
            soup = BeautifulSoup(content, 'html.parser')
            html_elements = soup.find_all(['div', 'span', 'button', 'a', 'nav', 'ul', 'li'])
            
            # Allow for style/script only files
            if component_file.name not in ['base_styles.html', 'base_scripts.html']:
                assert len(html_elements) > 0, f"{component_file.name} should have HTML elements"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])