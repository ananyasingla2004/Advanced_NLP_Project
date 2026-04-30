"""
Flask Web Server for Medical Chatbot
=====================================
REST API backend for the medical chatbot with CORS support for frontend integration.

Run this server with:
    python app.py

The server will be available at http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from medical_chatbot_api import MedicalChatbotAPI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend

# Initialize the chatbot API
try:
    chatbot_api = MedicalChatbotAPI('symptom_sentence_dataset_with_department.csv')
    logger.info("✓ Chatbot API initialized successfully")
except Exception as e:
    logger.error(f"✗ Failed to initialize chatbot: {e}")
    chatbot_api = None


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """Home endpoint - returns API information."""
    return jsonify({
        'status': 'running',
        'message': 'Medical Chatbot API Server',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/chat': 'Send a symptom query and get diagnosis',
            'GET /api/stats': 'Get chatbot statistics',
            'GET /api/health': 'Health check'
        }
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'chatbot_ready': chatbot_api is not None
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint.
    
    Request JSON:
    {
        "message": "I have a fever and cough",
        "language": "en" (optional)
    }
    
    Response JSON:
    {
        "status": "success",
        "predicted_disease": "Common Cold",
        "predicted_department": "General Medicine",
        "confidence": 0.85,
        "response": "...",
        "alternative_matches": [...]
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing "message" field in request'
            }), 400
        
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'status': 'error',
                'message': 'Message cannot be empty'
            }), 400
        
        if chatbot_api is None:
            return jsonify({
                'status': 'error',
                'message': 'Chatbot not initialized'
            }), 500
        
        # Process the query
        result = chatbot_api.analyze_symptoms(user_message)
        
        logger.info(f"Query processed: {user_message[:50]}...")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error processing your message: {str(e)}'
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get chatbot statistics."""
    try:
        if chatbot_api is None:
            return jsonify({
                'status': 'error',
                'message': 'Chatbot not initialized'
            }), 500
        
        stats = chatbot_api.get_statistics()
        return jsonify({
            'status': 'success',
            'data': stats
        })
    
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving statistics: {str(e)}'
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 Medical Chatbot API Server")
    print("="*60)
    print("Starting Flask server...")
    print("📍 Server running at: http://localhost:8000")
    print("💬 Chat endpoint: POST http://localhost:8000/api/chat")
    print("📊 Stats endpoint: GET http://localhost:8000/api/stats")
    print("="*60 + "\n")
    
    app.run(debug=True, port=8000, host='0.0.0.0')
