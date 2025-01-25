from flask import Flask, request, jsonify
import google.generativeai as ai
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import logging
from datetime import datetime
import json

# Configure the API Key
API_KEY = 'AIzaSyA_JwM_e6lVwiIkQy-B22Jat2yDhlpfN4c'
ai.configure(api_key=API_KEY)

# Set up logging
logging.basicConfig(filename='lightai_chat.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Initialize the Generative Model and Start Chat
class LightAIChat:
    def __init__(self, model_name="gemini-pro"):
        self.model = ai.GenerativeModel(model_name)
        self.chat = self.model.start_chat()

    def send_message(self, message):
        ai_response = self.chat.send_message(message)
        # Replace mentions of "Gemini" in responses
        cleaned_response = ai_response.text.replace("Gemini", "LightAI")
        # Conditionally replace "Google" with "LightAI" for specific contexts
        if any(keyword in ai_response.text.lower() for keyword in ["developed", "made you", "created you", "who created you", "did google create you", "trained you"]):
            cleaned_response = cleaned_response.replace("Google", "LightAI")
        return cleaned_response

# Add functionality to save chat history
class ChatHistory:
    def __init__(self):
        self.history = []

    def add_message(self, user, message):
        self.history.append({"user": user, "message": message, "timestamp": datetime.now().isoformat()})

    def save_to_file(self, filename="chat_history.json"):
        with open(filename, "w") as file:
            json.dump(self.history, file, indent=4)

# Instantiate the LightAIChat
lightai_chat = LightAIChat()
chat_history = ChatHistory()

# Define the Knowledge Base
knowledge_base = {
    "who are you": "I am LightAI, your AI assistant here to help you!",
    "what is your purpose": "My purpose is to assist and provide information to you.",
    "what is ai": "AI stands for Artificial Intelligence, which refers to systems designed to mimic human-like intelligence.",
    "bye": "Goodbye! It was nice chatting with you.",
    "who created you": "I was created by a visionary developer named Shreyan Dasari.",
    "who are your creators": "I am LightAI, a multi-modal model, brought to life by Shreyan Dasari."
}

# Threshold for fuzzy matching
FUZZY_MATCH_THRESHOLD = 80  # Matches above 80% similarity

# Fuzzy matching function
def get_fuzzy_match(user_input, knowledge_base):
    # Attempt to find the closest match in the knowledge base
    match, score = process.extractOne(user_input, knowledge_base.keys(), scorer=fuzz.ratio)
    if score >= FUZZY_MATCH_THRESHOLD:
        return match
    return None

# Function to handle user input and generate a response
def handle_user_input(user_input):
    matched_key = get_fuzzy_match(user_input, knowledge_base)
    if matched_key:
        response = knowledge_base[matched_key]
    else:
        response = lightai_chat.send_message(user_input)
    return response

# Define REST endpoint for chatbot
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

# Run Flask server
if __name__ == "__main__":
    app.run(port=5000)
