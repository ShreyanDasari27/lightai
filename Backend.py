from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as ai
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import logging
from datetime import datetime
import json
import os  # For accessing environment variables
from werkzeug.utils import secure_filename
import tempfile
import mimetypes

# For processing different file types
import PyPDF2
from PIL import Image
import pytesseract
import io

# Configure the API Key from environment variable
API_KEY = os.getenv("API_KEY")
ai.configure(api_key=API_KEY)

# Set up logging
logging.basicConfig(filename='lightai_chat.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize the Generative Model and Start Chat
class LightAIChat:
    def __init__(self, model_name="gemini-pro"):
        self.model = ai.GenerativeModel(model_name)
        self.chat = self.model.start_chat()

    def send_message(self, message, file_content=None):
        if file_content:
            # Modify this part based on how your API handles file content
            # For example, if the API accepts file content as context or a separate input
            # Here, we'll concatenate the file content with the message
            combined_message = f"{message}\n\nFile Content:\n{file_content}"
            ai_response = self.chat.send_message(combined_message)
        else:
            ai_response = self.chat.send_message(message)

        # Replace mentions of "Gemini" in responses
        cleaned_response = ai_response.text.replace("Gemini", "LightAI")
        # Conditionally replace "Google" with "LightAI" for specific contexts
        if any(keyword in ai_response.text.lower() for keyword in ["developed", "made you", "created you", "who created you", "did google create you", "trained you", "trained", "trained by", "made by"]):
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

# Function to extract text from different file types
def extract_text_from_file(file_storage):
    filename = secure_filename(file_storage.filename)
    file_extension = filename.rsplit('.', 1)[1].lower()

    try:
        if file_extension == 'txt':
            return file_storage.read().decode('utf-8')
        elif file_extension == 'pdf':
            reader = PyPDF2.PdfReader(file_storage)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        elif file_extension in {'png', 'jpg', 'jpeg', 'gif'}:
            image = Image.open(file_storage)
            text = pytesseract.image_to_string(image)
            return text
        else:
            return "Unsupported file type for text extraction."
    except Exception as e:
        logging.error(f"Error extracting text from file: {e}")
        return "Error processing the uploaded file."

# Function to handle user input and generate a response
def handle_user_input(user_input, file_content=None):
    matched_key = get_fuzzy_match(user_input, knowledge_base)
    if matched_key:
        response = knowledge_base[matched_key]
    else:
        response = lightai_chat.send_message(user_input, file_content=file_content)
    return response

# Define REST endpoint for chatbot
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = ""
        file_content = None

        if 'file' in request.files and request.files['file']:
            file = request.files['file']
            if file and allowed_file(file.filename):
                # Extract text from the uploaded file
                file_content = extract_text_from_file(file)
                # Optionally, save the file to a temporary location
                # with tempfile.NamedTemporaryFile(delete=True) as tmp:
                #     file.save(tmp.name)
            else:
                return jsonify({"error": "Unsupported file type."}), 400

        # Handle both form data and JSON
        if request.form:
            user_input = request.form.get("message", "")
        elif request.is_json:
            data = request.get_json()
            user_input = data.get("message", "")

        response = handle_user_input(user_input, file_content=file_content)
        chat_history.add_message("User", user_input if user_input else "Uploaded a file")
        chat_history.add_message("LightAI", response)
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error handling request: {e}")
        return jsonify({"error": "An error occurred processing your request."}), 500

# Default route for serving the HTML frontend
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# Run Flask server
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable
    app.run(host='0.0.0.0', port=port)
