// Chat interface handling
const API_URL = 'http://localhost:8000';
let conversationId = null;
let userId = `user_${Math.floor(Math.random() * 10000)}`;
let isProcessing = false;
let typingTimeout = null;

// Initialize chat interface
window.onload = async function() {
    await checkServerHealth();
    setupEventListeners();
    focusInput();
};

// Check server health
async function checkServerHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        updateServerStatus(data.status === 'ok' ? 'connected' : 'not-ready');
    } catch (error) {
        console.error('Error connecting to server:', error);
        updateServerStatus('error');
    }
}

// Update server connection status
function updateServerStatus(status) {
    const statusEl = document.getElementById('status');
    const statusContent = {
        'connected': `
            <span class="icon has-text-success">
                <i class="fas fa-check-circle"></i>
            </span>
            Connected to server
        `,
        'not-ready': `
            <span class="icon has-text-warning">
                <i class="fas fa-exclamation-triangle"></i>
            </span>
            Server is not ready
        `,
        'error': `
            <span class="icon has-text-danger">
                <i class="fas fa-times-circle"></i>
            </span>
            Failed to connect to server
        `
    };
    statusEl.innerHTML = statusContent[status] || statusContent['error'];
}

// Set up event listeners
function setupEventListeners() {
    const userInput = document.getElementById('user-input');
    userInput.addEventListener('keydown', handleInputKeydown);
    document.getElementById('send-button').addEventListener('click', sendMessage);
}

// Handle input keydown events
function handleInputKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

// Focus input field
function focusInput() {
    document.getElementById('user-input').focus();
}

// Send message to server
async function sendMessage() {
    if (isProcessing) return;
    
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const message = userInput.value.trim();
    
    if (!message) return;
    
    userInput.disabled = true;
    isProcessing = true;
    sendButton.classList.add('is-loading');
    
    // Clear any existing 'typing' indicators
    document.querySelectorAll('.bot-message.typing').forEach(el => el.remove());
    
    const userMessage = message;
    userInput.value = '';
    
    const userMessageId = addMessageToChat('user', userMessage);
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const botMessageId = addMessageToChat('bot', createTypingIndicator());
    
    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: userMessage,
                user_id: userId,
                conversation_id: conversationId
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!conversationId) {
            conversationId = data.conversation_id;
        }
        
        // Clean and enhance the response
        let enhancedResponse = data.response;
        
        // Remove common greetings and pleasantries
        enhancedResponse = enhancedResponse
            .replace(/^(hi|hello|hey|greetings)[\s,!]+/i, '')
            .replace(/how are you( doing)? today\??/i, '')
            .replace(/nice to (meet|see) you!?/i, '')
            .replace(/i('m| am) (happy|glad) to help( you)?!?/i, '')
            .replace(/is there anything else (i|we) can help you with\??/i, '')
            .replace(/please let (me|us) know if you have any (other |more )?questions!?/i, '')
            .trim();

        // Split into paragraphs and sentences for better formatting
        const paragraphs = enhancedResponse.split(/\n\n+/);
        let formattedParagraphs = [];
        let wordCount = 0;
        
        // Process paragraphs up to 900 words
        for (const para of paragraphs) {
            if (wordCount >= 900) break;
            
            const paraWords = para.split(/\s+/);
            const remainingWords = 900 - wordCount;
            
            if (paraWords.length <= remainingWords) {
                formattedParagraphs.push(para);
                wordCount += paraWords.length;
            } else {
                // Add partial paragraph to reach 900 words
                formattedParagraphs.push(paraWords.slice(0, remainingWords).join(' '));
                break;
            }
        }
        
        // Generate relevant follow-up questions
        const mainTopics = extractTopics(enhancedResponse);
        const followUps = generateFollowUpQuestions(mainTopics, userMessage);
        
        // Combine the formatted response
        enhancedResponse = formattedParagraphs.join('\n\n');
        if (followUps.length > 0) {
            enhancedResponse += '\n\nWould you like to know more about:\n' + followUps.join('\n');
        }
        
        replaceMessage(botMessageId, 'bot', enhancedResponse);
        
    } catch (error) {
        console.error('Error sending message:', error);
        const errorMessage = "I apologize, but I encountered an error. Could you please try asking your question again?";
        replaceMessage(botMessageId, 'bot', errorMessage);
    } finally {
        isProcessing = false;
        sendButton.classList.remove('is-loading');
        userInput.disabled = false;
        userInput.focus();
    }
}

// Create typing indicator HTML
function createTypingIndicator() {
    return `
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
}

// Create error message HTML
function createErrorMessage() {
    return "I apologize, but I encountered an error. Could you please try asking your question again?";
}

// Add message to chat
function addMessageToChat(sender, text) {
    const chatContainer = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    
    // Set proper classes and ensure layout
    messageDiv.className = sender === 'user' ? 'user-message' : 'bot-message';
    if (text.includes('typing-indicator')) {
        messageDiv.classList.add('typing');
    }
    
    // Create a unique ID for the message
    const messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    messageDiv.id = messageId;
    
    // Set the content safely
    const textContent = text;
    messageDiv.innerHTML = textContent;
    
    // Add to container and scroll
    chatContainer.appendChild(messageDiv);
    
    // Ensure proper scroll behavior
    requestAnimationFrame(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
        messageDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
    
    return messageId;
}

// Replace message content
function replaceMessage(messageId, sender, text) {
    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return;
    
    // Remove typing class if present
    messageDiv.classList.remove('typing');
    
    // Format the message
    const formattedText = formatMessage(text);
    messageDiv.innerHTML = formattedText;
    
    // Smooth scroll to show the message
    messageDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Extract main topics from the response
function extractTopics(text) {
    const sentences = text.split(/[.!?]+/);
    const topics = new Set();
    
    // Keywords that might indicate important topics
    const keywords = ['is', 'are', 'has', 'have', 'can', 'includes', 'contains', 'features', 'consists'];
    const excludedWords = ['i', 'you', 'we', 'they', 'it', 'this', 'that', 'these', 'those'];
    
    sentences.forEach(sentence => {
        keywords.forEach(keyword => {
            const pattern = new RegExp(`\\b([\\w\\s]{3,30})\\s+${keyword}\\b`, 'i');
            const match = sentence.match(pattern);
            if (match) {
                const topic = match[1].trim().toLowerCase();
                if (!excludedWords.some(word => topic === word)) {
                    topics.add(topic);
                }
            }
        });
    });
    
    return Array.from(topics);
}

// Generate relevant follow-up questions
function generateFollowUpQuestions(topics, originalQuery) {
    const questions = new Set();
    const queryWords = originalQuery.toLowerCase().split(/\s+/);
    
    // Extract main subject from the query
    const subjectIndex = Math.max(
        queryWords.indexOf('about'),
        queryWords.indexOf('regarding'),
        queryWords.indexOf('concerning')
    );
    const mainSubject = subjectIndex !== -1 ? 
        queryWords.slice(subjectIndex + 1).join(' ') : 
        queryWords.join(' ');
    
    // Question templates
    const templates = [
        topic => `• How does ${topic} work?`,
        topic => `• What are the key features of ${topic}?`,
        topic => `• Can you explain more about ${topic}?`,
        topic => `• What are the different types of ${topic}?`,
        topic => `• What are the best practices for ${topic}?`
    ];
    
    // Generate questions from topics
    topics.slice(0, 3).forEach(topic => {
        const template = templates[Math.floor(Math.random() * templates.length)];
        questions.add(template(topic));
    });
    
    // If we have fewer than 3 questions, add some based on the main subject
    if (questions.size < 3) {
        while (questions.size < 3) {
            const template = templates[Math.floor(Math.random() * templates.length)];
            questions.add(template(mainSubject));
        }
    }
    
    return Array.from(questions);
}

// Format message text with proper HTML
function formatMessage(text) {
    // First escape HTML
    let formatted = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    
    // Format code blocks
    formatted = formatted.replace(/```(\w*)([\s\S]*?)```/g, (match, language, code) => 
        `<pre><code class="language-${language || 'plaintext'}">${code.trim()}</code></pre>`
    );
    
    // Format inline code
    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Split into paragraphs
    const paragraphs = formatted.split(/\n\n+/);
    
    // Process each paragraph
    formatted = paragraphs.map(para => {
        para = para.trim();
        if (!para) return '';
        
        // Don't wrap code blocks in paragraphs
        if (para.startsWith('<pre>')) return para;
        
        // Replace single newlines with <br>
        para = para.replace(/\n/g, '<br>');
        
        // Wrap in paragraph tags
        return `<p>${para}</p>`;
    }).join('\n');
    
    return formatted || text;
}
