#!/usr/bin/env python3
"""Debug server with detailed network binding information."""

import sys
import os
import socket
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_port_binding(host, port):
    """Test if we can bind to a specific host:port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(1)
        print(f"✅ Successfully bound to {host}:{port}")
        sock.close()
        return True
    except Exception as e:
        print(f"❌ Failed to bind to {host}:{port} - {e}")
        return False

def main():
    print("🔍 Network Debugging Information:")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")
    
    # Test different host configurations
    hosts_to_test = [
        ("127.0.0.1", 8888),
        ("localhost", 8889), 
        ("0.0.0.0", 8890)
    ]
    
    print("\n🧪 Testing port binding capabilities:")
    working_config = None
    
    for host, port in hosts_to_test:
        if test_port_binding(host, port):
            working_config = (host, port)
            break
    
    if not working_config:
        print("\n❌ Cannot bind to any host:port combination. This indicates a system-level networking issue.")
        print("Possible causes:")
        print("- macOS Firewall blocking Python")
        print("- Security software interference") 
        print("- Network configuration issues")
        return
    
    print(f"\n✅ Found working configuration: {working_config[0]}:{working_config[1]}")
    print("Now attempting to start the main application...")
    
    try:
        import uvicorn
        from web.main import app
        
        host, port = working_config
        print(f"🚀 Starting server on http://{host}:{port}")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Make sure virtual environment is activated")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")

if __name__ == "__main__":
    main()