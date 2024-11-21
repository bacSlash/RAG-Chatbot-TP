class ChatInterface {
    constructor() {
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.micButton = document.getElementById('micButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.sessionId = Date.now().toString();
        this.chatHistory = [];

        this.initializeEventListeners();
    }

    initializeEventListeners() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.micButton.addEventListener('click', () => this.toggleRecording());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
    }

    async initializeMediaRecorder() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);

            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/mp3' });
                this.audioChunks = [];
                await this.sendAudioToServer(audioBlob);
            };
        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Error accessing microphone. Please ensure microphone permissions are granted.');
        }
    }

    async toggleRecording() {
        if (!this.mediaRecorder) {
            await this.initializeMediaRecorder();
        }

        if (this.isRecording) {
            this.mediaRecorder.stop();
            this.micButton.classList.remove('recording');
        } else {
            this.mediaRecorder.start();
            this.micButton.classList.add('recording');
        }
        this.isRecording = !this.isRecording;
    }

    async sendAudioToServer(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob);

        try {
            const response = await fetch('/speech-to-text', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.text) {
                this.messageInput.value = data.text;
                this.sendMessage();
            }
        } catch (error) {
            console.error('Error sending audio to server:', error);
            this.addMessage('Error processing audio. Please try again.', 'bot');
        }
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        this.addMessage(message, 'user');
        this.messageInput.value = '';

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    chat_history: this.chatHistory,
                    question: message
                })
            });

            const data = await response.json();
            if (data.result) {
                this.addMessage(data.result, 'bot');
            } else {
                throw new Error('Invalid response from server');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('Sorry, there was an error processing your message.', 'bot');
        }
    }

    addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
    
        // Add user or bot logo
        const logo = document.createElement('img');
        if (type === 'user') {
            logo.src = '/static/images/user-logo.jpg'; // Replace with your user logo path
            logo.alt = 'User';
        } else {
            logo.src = '/static/images/logo.jpg'; // TPBot logo
            logo.alt = 'TPBot';
        }
    
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = content;
    
        messageDiv.appendChild(logo);
        messageDiv.appendChild(messageContent);
    
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    
        // Update chat history
        this.chatHistory.push({
            role: type === 'user' ? 'user' : 'assistant',
            content: content
        });
    }    

    async speakMessage(text) {
        try {
            const response = await fetch('/text-to-speech', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            if (!response.ok) throw new Error('Text-to-speech request failed');

            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            audio.play();
        } catch (error) {
            console.error('Error in text-to-speech:', error);
            alert('Error playing audio. Please try again.');
        }
    }

    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        this.chatMessages.appendChild(typingDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        return typingDiv;
    }

    removeTypingIndicator(element) {
        if (element && element.parentNode) {
            element.parentNode.removeChild(element);
        }
    }
}

// Initialize chat interface when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatInterface = new ChatInterface();
});
