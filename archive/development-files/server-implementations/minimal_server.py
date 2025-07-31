#!/usr/bin/env python3
"""Minimal server for testing network connectivity."""

import socketserver
import http.server
import sys

class TestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        message = """
        <html>
        <head><title>Network Test</title></head>
        <body>
        <h1>✅ Network Connection Working!</h1>
        <p>This confirms that basic HTTP serving works on this system.</p>
        <p>The issue may be with the FastAPI application startup or port binding.</p>
        </body>
        </html>
        """
        self.wfile.write(message.encode())

def main():
    PORT = 8888
    try:
        with socketserver.TCPServer(("0.0.0.0", PORT), TestHandler) as httpd:
            print(f"✅ Minimal test server running at http://localhost:{PORT}")
            print("Press Ctrl+C to stop")
            httpd.serve_forever()
    except Exception as e:
        print(f"❌ Failed to start test server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()