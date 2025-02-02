from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as ai
from fuzzywuzzy import fuzz, process
import logging
from datetime import datetime
import json
import os
import base64  # Built-in module for encoding image data

# Configure the API key from environment variables
API_KEY = os.getenv("API_KEY")
ai.configure(api_key=API_KEY)

# Set up logging
logging.basicConfig(filename='lightai_chat.log', level=logging.INFO, format='%(asctime)s - %(message)s')

app = Flask(__name__, static_folder='.', static_url_path='')

class LightAIChat:
    def __init__(self, model_name="gemini-pro"):
        self.model = ai.GenerativeModel(model_name)
        self.chat = self.model.start_chat()

    def send_message(self, message):
        ai_response = self.chat.send_message(message)
        # Replace mentions of "Gemini" with "LightAI"
        cleaned_response = ai_response.text.replace("Gemini", "LightAI")
        if any(keyword in ai_response.text.lower() for keyword in [
            "developed", "made you", "created you", "who created you",
            "did google create you", "trained you", "trained", "trained by", "made by"
        ]):
            cleaned_response = cleaned_response.replace("Google", "LightAI")
        return cleaned_response

class ChatHistory:
    def __init__(self):
        self.history = []

    def add_message(self, user, message):
        self.history.append({"user": user, "message": message, "timestamp": datetime.now().isoformat()})

    def save_to_file(self, filename="chat_history.json"):
        with open(filename, "w") as file:
            json.dump(self.history, file, indent=4)

# Instantiate the chat and history objects
lightai_chat = LightAIChat()
chat_history = ChatHistory()

# Define the knowledge base
knowledge_base = {
    "who are you": "I am LightAI, your AI assistant here to help you!",
    "what is your purpose": "My purpose is to assist and provide information to you.",
    "what is ai": "AI stands for Artificial Intelligence, which refers to systems designed to mimic human-like intelligence.",
    "bye": "Goodbye! It was nice chatting with you.",
    "who created you": "I was created by a visionary developer named Shreyan Dasari.",
    "who are your creators": "I am LightAI, a multi-modal model, brought to life by Shreyan Dasari."
}

# Fuzzy matching threshold
FUZZY_MATCH_THRESHOLD = 80

def get_fuzzy_match(user_input, knowledge_base):
    match, score = process.extractOne(user_input, knowledge_base.keys(), scorer=fuzz.ratio)
    if score >= FUZZY_MATCH_THRESHOLD:
        return match
    return None

def handle_user_input(user_input):
    matched_key = get_fuzzy_match(user_input, knowledge_base)
    if matched_key:
        response = knowledge_base[matched_key]
    else:
        response = lightai_chat.send_message(user_input)
    return response

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get("message", "")
        response = handle_user_input(user_input)
        chat_history.add_message("User", user_input)
        chat_history.add_message("LightAI", response)
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error handling request: {e}")
        return jsonify({"error": "An error occurred processing your request."}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file."}), 400

        # Process image files by encoding them in base64 and prompting analysis
        if file.content_type.startswith('image/'):
            image_bytes = file.read()
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            text = f"Please analyze the following image encoded in base64: {encoded_image}"
        else:
            content = file.read()
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = f"File '{file.filename}' received, but it appears to be a binary file."

        response = handle_user_input(text)
        chat_history.add_message("User (file)", text)
        chat_history.add_message("LightAI", response)
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error processing file upload: {e}")
        return jsonify({"error": "Error processing file upload."}), 500

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
