/**
 * Medical Chatbot Frontend - JavaScript
 * ======================================
 * Handles all frontend interactions and API communication
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000';
const API_CHAT_ENDPOINT = `${API_BASE_URL}/api/chat`;
const API_HEALTH_ENDPOINT = `${API_BASE_URL}/api/health`;

// DOM Elements
const messagesContainer = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const chatForm = document.getElementById('chatForm');
const loadingModal = document.getElementById('loadingModal');
const errorModal = document.getElementById('errorModal');
const errorMessage = document.getElementById('errorMessage');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// Application state
let isConnected = false;
let isLoading = false;
let messageHistory = [];

/**
 * Initialize the application
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Medical Chatbot Frontend Initialized');
    
    // Setup event listeners
    chatForm.addEventListener('submit', handleSendMessage);
    
    // Check connection
    checkConnection();
    
    // Check connection every 5 seconds
    setInterval(checkConnection, 5000);
    
    // Focus input on load
    userInput.focus();
});

/**
 * Check if the backend server is running
 */
async function checkConnection() {
    try {
        const response = await fetch(API_HEALTH_ENDPOINT, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            if (!isConnected) {
                isConnected = true;
                updateStatusIndicator(true);
                console.log('✓ Connected to backend server');
            }
        } else {
            handleConnectionError();
        }
    } catch (error) {
        handleConnectionError();
    }
}

/**
 * Update status indicator
 */
function updateStatusIndicator(connected) {
    if (connected) {
        statusDot.classList.add('connected');
        statusText.textContent = 'Connected';
        sendButton.disabled = false;
    } else {
        statusDot.classList.remove('connected');
        statusText.textContent = 'Disconnected';
        sendButton.disabled = true;
    }
}

/**
 * Handle connection error
 */
function handleConnectionError() {
    if (isConnected) {
        isConnected = false;
        updateStatusIndicator(false);
        console.warn('✗ Connection lost to backend server');
    }
}

/**
 * Handle form submission
 */
async function handleSendMessage(e) {
    e.preventDefault();
    
    const message = userInput.value.trim();
    
    if (!message) {
        return;
    }
    
    if (!isConnected) {
        showError('Server not connected. Please ensure the backend server is running.');
        return;
    }
    
    // Add user message to UI
    addMessageToUI(message, 'user');
    
    // Clear input
    userInput.value = '';
    
    // Store in history
    messageHistory.push({ role: 'user', content: message });
    
    // Process the message
    await sendMessageToAPI(message);
}

/**
 * Send message to the backend API
 */
async function sendMessageToAPI(message) {
    try {
        // Show loading modal
        showLoadingModal(true);
        isLoading = true;
        sendButton.disabled = true;
        
        // Make API call
        const response = await fetch(API_CHAT_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                language: null // Auto-detect
            })
        });
        
        // Hide loading modal
        showLoadingModal(false);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || 'Server error');
        }
        
        const data = await response.json();
        
        // Handle the response
        handleChatResponse(data);
        
    } catch (error) {
        showLoadingModal(false);
        console.error('Error sending message:', error);
        showError(`Failed to process message: ${error.message}`);
        addMessageToUI(
            '❌ Sorry, I encountered an error processing your message. Please try again.',
            'bot'
        );
    } finally {
        isLoading = false;
        sendButton.disabled = false;
        userInput.focus();
    }
}

/**
 * Handle API response
 */
function handleChatResponse(data) {
    if (data.status === 'error') {
        addMessageToUI(`❌ ${data.message}`, 'bot');
        return;
    }
    
    if (data.status === 'no_match') {
        const botMessage = `
            <div class="response-section">
                <p>I couldn't find a clear match for your symptoms. This might be because:</p>
                <ul style="margin-top: 8px; margin-left: 20px;">
                    <li>The symptom description is too vague</li>
                    <li>The combination is unusual</li>
                    <li>Multiple conditions could match</li>
                </ul>
                <p style="margin-top: 8px;">Please try describing your symptoms in more detail, or consult a healthcare professional.</p>
            </div>
        `;
        addMessageToUI(botMessage, 'bot', true);
        messageHistory.push({ role: 'bot', content: botMessage });
        return;
    }
    
    // Build the response message
    let botMessage = '<div style="display: flex; flex-direction: column; gap: 10px;">';
    
    if (data.predicted_disease) {
        botMessage += `
            <div class="response-section">
                <div class="response-section-title">🔍 Predicted Disease/Condition</div>
                <div class="response-value"><strong>${data.predicted_disease}</strong></div>
            </div>
        `;
    }
    
    if (data.predicted_department) {
        botMessage += `
            <div class="response-section">
                <div class="response-section-title">🏥 Recommended Department</div>
                <div class="response-value"><strong>${data.predicted_department}</strong></div>
            </div>
        `;
    }
    
    if (data.confidence !== undefined) {
        const confidencePercent = Math.round(data.confidence * 100);
        botMessage += `
            <div class="response-section">
                <div class="response-section-title">📊 Confidence Level</div>
                <div class="confidence-bar">
                    <div class="confidence-bar-fill">
                        <div class="confidence-bar-progress" style="width: ${confidencePercent}%;"></div>
                    </div>
                    <div class="confidence-percentage">${confidencePercent}%</div>
                </div>
            </div>
        `;
    }
    
    if (data.response) {
        botMessage += `
            <div class="response-section">
                <div class="response-section-title">💬 Additional Information</div>
                <div class="response-value">${data.response}</div>
            </div>
        `;
    }
    
    if (data.alternative_matches && data.alternative_matches.length > 0) {
        botMessage += `
            <div class="alternatives-section">
                <div class="alternatives-title">🔄 Alternative Matches</div>
        `;
        
        data.alternative_matches.forEach(alt => {
            const altConfidence = Math.round(alt.confidence * 100);
            botMessage += `
                <div class="alternative-item">
                    <span class="alternative-disease">
                        ${alt.disease}
                        <span style="color: #999; font-size: 0.85em;"> (${alt.department})</span>
                    </span>
                    <span class="alternative-confidence">${altConfidence}%</span>
                </div>
            `;
        });
        
        botMessage += '</div>';
    }
    
    botMessage += '</div>';
    
    // Add bot message to UI
    addMessageToUI(botMessage, 'bot', true);
    messageHistory.push({ role: 'bot', content: botMessage });
}

/**
 * Add message to the UI
 */
function addMessageToUI(message, sender, isHTML = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isHTML) {
        contentDiv.innerHTML = message;
    } else {
        contentDiv.textContent = message;
    }
    
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    
    // Auto-scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Show/hide loading modal
 */
function showLoadingModal(show) {
    if (show) {
        loadingModal.classList.remove('hidden');
    } else {
        loadingModal.classList.add('hidden');
    }
}

/**
 * Show error modal
 */
function showError(message) {
    errorMessage.textContent = message;
    errorModal.classList.remove('hidden');
}

/**
 * Close error modal
 */
function closeErrorModal() {
    errorModal.classList.add('hidden');
}

/**
 * Close error modal when clicking outside
 */
document.addEventListener('click', (e) => {
    if (e.target === errorModal) {
        closeErrorModal();
    }
});

/**
 * Keyboard shortcuts
 */
document.addEventListener('keydown', (e) => {
    // Ctrl+Enter or Cmd+Enter to send message
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (userInput.value.trim()) {
            handleSendMessage(new Event('submit'));
        }
    }
    
    // Escape to close error modal
    if (e.key === 'Escape') {
        closeErrorModal();
    }
});

console.log('✅ Script loaded successfully');
