@echo off
REM Medical Chatbot Startup Script for Windows
REM ============================================

echo.
echo 🏥 Medical Chatbot - Startup Script
echo ====================================
echo.

REM Check if port 8000 is in use
netstat -ano | findstr :8000 >nul
if %errorlevel% equ 0 (
    echo ⚠️  Port 8000 is already in use!
    echo    The server might be running. Try:
    echo    1. Kill the process: taskkill /PID [process_id] /F
    echo    2. Or use a different port in app.py
    pause
    exit /b 1
)

REM Start Flask server
echo 🚀 Starting Flask server...
start cmd /k "python app.py"
echo ✓ Server started
echo.

REM Wait for server to start
echo ⏳ Waiting for server to be ready...
timeout /t 3 /nobreak

REM Check if server is responding
timeout /t 2 /nobreak

REM Open frontend in browser
echo 🌐 Opening frontend in browser...
set "FRONTEND_URL=file:///%cd%\index.html"
echo    URL: %FRONTEND_URL%
echo.

start "" "%FRONTEND_URL%"

echo.
echo ====================================
echo ✅ Chatbot is ready!
echo ====================================
echo.
echo 📍 Server URL:     http://localhost:8000
echo 💬 Chat API:       http://localhost:8000/api/chat
echo 🌐 Frontend URL:   file:///%cd%\index.html
echo.
echo Press Ctrl+C to stop the server
echo.
pause
