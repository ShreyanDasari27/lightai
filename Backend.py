from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as ai
from fuzzywuzzy import fuzz, process
import logging
from datetime import datetime
import json
import os
import base64
import io
from PIL import Image  # For image processing
import requests       # For external API calls

# Configure the API key for generative AI from environment variables
API_KEY = os.getenv("API_KEY")
ai.configure(api_key=API_KEY)

# Configure DeepAI API key (free alternative for image tagging)
DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY")

# Set up logging
logging.basicConfig(filename='lightai_chat.log', level=logging.INFO, format='%(asctime)s - %(message)s')

app = Flask(__name__, static_folder='.', static_url_path='')
# Limit upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

class LightAIChat:
    def __init__(self, model_name="gemini-pro"):
        self.model = ai.GenerativeModel(model_name)
        self.chat = self.model.start_chat()

    def send_message(self, message):
        try:
            ai_response = self.chat.send_message(message)
        except Exception as e:
            logging.error(f"Error from generative model: {e}")
            return "Sorry, I couldn't process the request due to an internal error."
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
        self.history.append({
            "user": user,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

    def save_to_file(self, filename="chat_history.json"):
        with open(filename, "w") as file:
            json.dump(self.history, file, indent=4)

# Instantiate the chat and history objects
lightai_chat = LightAIChat()
chat_history = ChatHistory()

# Define a simple knowledge base for common queries.
knowledge_base = {
    "who are you": "I am LightAI, your AI assistant here to help you!",
    "what is your purpose": "My purpose is to assist and provide information to you.",
    "what is ai": "AI stands for Artificial Intelligence, which refers to systems designed to mimic human-like intelligence.",
    "bye": "Goodbye! It was nice chatting with you.",
    "who created you": "I was created by a visionary developer named Shreyan Dasari.",
    "who are your creators": "I am LightAI, a multi-modal model, brought to life by Shreyan Dasari."
}

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

def process_image_with_deepai(image_bytes):
    """
    Uses DeepAI's Image Tagging API to get tags for the image.
    Returns a description string or None if unsuccessful.
    """
    try:
        url = "https://api.deepai.org/api/image-tagging"
        headers = {'api-key': DEEPAI_API_KEY}
        files = {'image': image_bytes}
        r = requests.post(url, files=files, headers=headers)
        result = r.json()
        if 'output' in result and 'tags' in result['output']:
            tags = result['output']['tags']
            top_tags = [tag['tag'] for tag in tags if float(tag.get('confidence', 0)) > 0.5][:5]
            if top_tags:
                return "Image contains: " + ", ".join(top_tags) + "."
        return None
    except Exception as e:
        logging.error(f"Error calling DeepAI API: {e}")
        return None

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
        logging.error(f"Error handling /chat request: {e}")
        return jsonify({"error": "An error occurred processing your request."}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file."}), 400

        if file.content_type.startswith('image/'):
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            # Resize if image width exceeds 800 pixels.
            max_width = 800
            if image.width > max_width:
                ratio = max_width / image.width
                new_size = (max_width, int(image.height * ratio))
                image = image.resize(new_size, Image.ANTIALIAS)
                buf = io.BytesIO()
                image_format = image.format if image.format else "PNG"
                image.save(buf, format=image_format)
                image_bytes = buf.getvalue()
            # Try using DeepAI API for a description.
            deepai_description = process_image_with_deepai(image_bytes)
            if deepai_description:
                text = deepai_description
            else:
                # Fallback: use base64-encoded image in a refined prompt.
                encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                text = (f"The following is an image encoded in base64. "
                        f"Please provide a detailed and accurate description of its content:\n\n"
                        f"{encoded_image}")
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
        logging.error(f"Error processing /upload request: {e}")
        return jsonify({"error": "Error processing file upload."}), 500

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)