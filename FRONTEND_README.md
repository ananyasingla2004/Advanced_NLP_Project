# Medical Chatbot Frontend Setup & Run Guide

## 📋 Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start Backend Server
```bash
python3 app.py
```

You'll see:
```
🏥 Medical Chatbot API Server
📍 Server running at: http://localhost:8000
```

**Keep this terminal running.**

### Step 3: Open Frontend (in another terminal)
```bash
open index.html
```

Or use Python server:
```bash
python3 -m http.server 8080
# Then visit: http://localhost:8080
```

---

## 🚀 Automated Startup (Easier)

**Mac/Linux:**
```bash
./startup.sh
```

**Windows:**
```bash
startup.bat
```

Both scripts automatically start the server and open the frontend in your browser.

---

## 💬 How to Use

1. **Backend loads** - BERT model initializes (~20-30 seconds on first run)
2. **Frontend opens** - Chat interface appears
3. **Type symptoms** - e.g., "I have fever and cough"
4. **Get results** - See disease prediction, department, and confidence score

---

## 🔍 Results Explained

- **Predicted Disease** - AI's best match for your symptoms
- **Department** - Which medical specialist to visit
- **Confidence** - Prediction accuracy (0-100%)
- **Alternatives** - Other possible conditions

---

## 🌐 Access Points

**Local:**
- Backend API: `http://localhost:8000`
- Frontend: `file:///path/to/index.html`

**From other devices (same WiFi):**
- `http://10.1.18.134:8000`

---

## 🛑 Stop Server

Press `Ctrl+C` in the terminal running the Flask server.

---

## 🐛 Troubleshooting

**Port 8000 already in use:**
```bash
lsof -i :8000 | awk 'NR>1 {print $2}' | xargs kill -9
```

**Module not found:**
```bash
pip install -r requirements.txt
```

**First query is slow:**
- Normal - BERT model loads on first request (5-10 seconds)
- Subsequent queries are faster

---

## 📁 Files

| File | Purpose |
|------|---------|
| `app.py` | Flask backend API |
| `index.html` | Web interface |
| `style.css` | Styling |
| `script.js` | Frontend logic |
| `startup.sh` | Auto-launcher (Mac/Linux) |
| `startup.bat` | Auto-launcher (Windows) |

---

## ✅ That's It!

Your medical chatbot is ready to use. 🏥💬
