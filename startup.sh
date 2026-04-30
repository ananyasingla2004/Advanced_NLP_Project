#!/bin/bash

# Medical Chatbot Startup Script
# ==============================
# This script starts the Flask server and opens the frontend in your browser

echo "🏥 Medical Chatbot - Startup Script"
echo "===================================="
echo ""

# Check if Flask server is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8000 is already in use!"
    echo "   The server might be running. Try:"
    echo "   1. Kill the process: lsof -ti:8000 | xargs kill -9"
    echo "   2. Or use a different port in app.py"
    exit 1
fi

# Start Flask server in background
echo "🚀 Starting Flask server..."
python app.py &
SERVER_PID=$!
echo "✓ Server started (PID: $SERVER_PID)"
echo ""

# Wait for server to start
echo "⏳ Waiting for server to be ready..."
sleep 3

# Check if server is running
if ! lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "❌ Failed to start server. Check if Python and dependencies are installed."
    echo ""
    echo "Install dependencies with: pip install -r requirements.txt"
    exit 1
fi

echo "✓ Server is ready!"
echo ""

# Open frontend in browser
FRONTEND_URL="file://$(pwd)/index.html"
echo "🌐 Opening frontend in browser..."
echo "   URL: $FRONTEND_URL"
echo ""

# Detect OS and open accordingly
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open "$FRONTEND_URL"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open "$FRONTEND_URL" 2>/dev/null || echo "Please open $FRONTEND_URL in your browser"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows
    start "$FRONTEND_URL"
else
    echo "Please open $FRONTEND_URL in your browser"
fi

echo ""
echo "===================================="
echo "✅ Chatbot is ready!"
echo "===================================="
echo ""
echo "📍 Server URL:     http://localhost:8000"
echo "💬 Chat API:       http://localhost:8000/api/chat"
echo "🌐 Frontend URL:   file://$(pwd)/index.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Keep script running
wait $SERVER_PID
