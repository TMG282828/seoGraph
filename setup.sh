#!/bin/bash
echo "🔧 Setting up SEO Content Knowledge Graph System..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
    echo "🚀 Starting the application..."
    python main.py
else
    echo "❌ Failed to install dependencies"
    echo "💡 Please check your requirements.txt file"
    exit 1
fi