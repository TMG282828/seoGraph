#!/usr/bin/env python3
"""Simple test server to verify network connectivity."""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

class TestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"""
        <html>
        <body>
        <h1>Test Server Working!</h1>
        <p>This confirms the network connection is working.</p>
        <p>You can access the main application once it's properly started.</p>
        </body>
        </html>
        """)

if __name__ == "__main__":
    PORT = 8080
    with socketserver.TCPServer(("", PORT), TestHandler) as httpd:
        print(f"Test server running at http://localhost:{PORT}")
        httpd.serve_forever()