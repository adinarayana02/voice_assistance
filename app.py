import io
import logging
import requests
from flask import Flask, render_template, request, jsonify, send_file
import pyttsx3
import speech_recognition as sr

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Groq API configuration
API_KEY = "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"
MODEL = "llama3-8b-8192"

# Conversational context storage
conversation_history = []
MAX_CONVERSATION_HISTORY = 5


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        # Check if audio file is in the request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file'}), 400

        # Get the audio file
        audio_file = request.files['audio']

        # Use SpeechRecognition to transcribe
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        return jsonify({'transcription': text})

    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio'}), 400
    except sr.RequestError as e:
        logger.error(f"Speech recognition error: {e}")
        return jsonify({'error': 'Speech recognition service error'}), 500
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({'error': 'Unexpected error during transcription'}), 500

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        user_text = request.json.get('text', '').strip()

        if not user_text:
            return jsonify({'error': 'No input text provided'}), 400

        # Manage conversation history
        conversation_history.append({"role": "user", "content": user_text})
        if len(conversation_history) > MAX_CONVERSATION_HISTORY:
            conversation_history.pop(0)

        # System prompt with Telugu teaching style
        messages = [
            {
                "role": "system",
                "content": """You are a friendly Telugu teacher teaching English to 5th class students.
                Use simple English with Telugu-style sentence structure. Add occasional Telugu words 
                like 'అవును' (yes), 'సరే' (okay), 'అర్థమైందా' (understood?). Speak like a Telugu 
                teacher naturally would."""
            }
        ] + conversation_history

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 300,
                }
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"API request error: {e}")
            return jsonify({'error': 'Failed to connect to AI service'}), 500

        try:
            response_data = response.json()
            ai_response = response_data['choices'][0]['message']['content'].strip()
            
            # Process response with Telugu accent and patterns
            ai_response = format_telugu_style_english(ai_response)
            
        except (KeyError, IndexError) as e:
            logger.error(f"Response parsing error: {e}")
            return jsonify({'error': 'Failed to parse AI response'}), 500

        conversation_history.append({"role": "assistant", "content": ai_response})

        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            
            # Find best voice for Telugu accent
            best_voice = None
            # First priority: South Indian English voice
            for voice in voices:
                if any(keyword in voice.name.lower() for keyword in ['indian', 'tamil', 'telugu', 'south']):
                    best_voice = voice
                    break
            
            # Second priority: Any Indian female voice
            if not best_voice:
                for voice in voices:
                    if 'indian' in voice.name.lower() and ('female' in voice.name.lower()):
                        best_voice = voice
                        break

            if best_voice:
                engine.setProperty('voice', best_voice.id)
                logger.info(f"Using voice: {best_voice.name}")
            
            # Telugu accent-optimized settings
            engine.setProperty('rate', 115)      # Very slow pace typical of Telugu English
            engine.setProperty('volume', 0.85)   # Comfortable volume
            engine.setProperty('pitch', 1.1)     # Slightly higher pitch for Telugu accent
            
            # Add Telugu-style pauses and emphasis
            ai_response_with_accent = add_telugu_accent_patterns(ai_response)
            
            audio_file = "response.mp3"
            engine.save_to_file(ai_response_with_accent, audio_file)
            engine.runAndWait()

            return jsonify({
                'text': ai_response,
                'audio_url': f'/download_audio/{audio_file}'
            })

        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            return jsonify({
                'text': ai_response,
                'audio_url': None,
                'error': 'Failed to generate audio'
            })

    except Exception as e:
        logger.error(f"Unexpected error in generate_response: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

def format_telugu_style_english(text):
    """Format text to match Telugu English patterns."""
    # Split into very short phrases
    sentences = text.split('. ')
    telugu_style_sentences = []
    
    for sentence in sentences:
        words = sentence.split()
        # Break into smaller phrases (Telugu style)
        if len(words) > 5:
            phrases = [words[i:i + 4] for i in range(0, len(words), 4)]
            for phrase in phrases:
                telugu_style_sentences.append(' '.join(phrase) + '.')
        else:
            telugu_style_sentences.append(sentence + '.')
    
    # Apply Telugu English word patterns
    text = ' '.join(telugu_style_sentences)
    telugu_patterns = {
        # Common Telugu English patterns
        "isn't it?": "no?",
        "isn't that right?": "correct no?",
        "do you understand?": "artham ayyinda?",
        "right?": "kadaa?",
        "look here": "ikkada choodandi",
        "listen carefully": "jagratha ga vinandi",
        # Simplified words common in Telugu English
        "difficult": "kastam",
        "easy": "easy ga",
        "very good": "chala bagundi",
        "understand": "artham chesuko",
        "think": "alochinchu",
        "slowly": "mellaga",
        "carefully": "jagratha ga"
    }
    
    for eng_pattern, telugu_pattern in telugu_patterns.items():
        text = text.replace(eng_pattern, telugu_pattern)
    
    return text

def add_telugu_accent_patterns(text):
    """Add speech patterns typical of Telugu English accent."""
    # Add characteristic pauses of Telugu English
    text = text.replace('. ', '... ... ')  # Longer pauses between sentences
    text = text.replace(', ', '... , ')    # Medium pauses after commas
    
    # Add emphasis patterns common in Telugu English
    emphasis_words = [
        'correct', 'wrong', 'good', 'bad', 'important', 
        'remember', 'look', 'listen', 'understand', 'carefully'
    ]
    
    for word in emphasis_words:
        # Add typical Telugu emphasis pattern
        text = text.replace(f' {word} ', f' ... {word} ... ')
    
    # Add stress on question words (typical in Telugu English)
    question_words = ['what', 'why', 'how', 'when', 'where', 'who']
    for word in question_words:
        text = text.replace(f' {word} ', f' ... {word}... ')
    
    # Add typical Telugu filler sounds
    text = text.replace('um ', 'ahh ')
    text = text.replace('uh ', 'ohh ')
    
    # Emphasize numbers and dates (common in Telugu teaching)
    import re
    numbers = re.findall(r'\d+', text)
    for number in numbers:
        text = text.replace(number, f'... {number} ...')
    
    return text

def simplify_math_terms(text):
    """Simplify mathematical terms using Telugu-style explanations."""
    math_terms = {
        "addition": "koodithe",
        "subtraction": "teesithe",
        "multiplication": "gunisthe",
        "division": "bhagisthe",
        "equals": "samaanam",
        "plus": "plus",
        "minus": "minus"
    }
    
    for eng_term, telugu_term in math_terms.items():
        text = text.replace(eng_term, telugu_term)
    
    return text

def simplify_for_kids(text):
    """Simplify the language for primary school children."""
    import re
    replacements = {
        "articulate": "clear",
        "informative": "helpful",
        "responses": "answers",
        "conversation": "chat",
        "assistant": "helper",
        "system": "computer",
    }
    for complex_word, simple_word in replacements.items():
        text = re.sub(rf'\b{complex_word}\b', simple_word, text, flags=re.IGNORECASE)
    sentences = text.split('. ')
    simplified_sentences = [s for s in sentences if len(s.split()) <= 15]
    return '. '.join(simplified_sentences)

def add_fillers(text):
    """Add natural fillers like 'hmm' and 'ahh' to text."""
    import random

    fillers = ["hmm", "ahh", "you know", "let's see", "I mean", "well"]
    sentences = text.split('. ')
    enhanced_sentences = []

    for sentence in sentences:
        if random.random() < 0.3:  # 30% chance to add a filler
            filler = random.choice(fillers)
            enhanced_sentences.append(f"{filler}, {sentence}")
        else:
            enhanced_sentences.append(sentence)

    return '. '.join(enhanced_sentences)

@app.route('/download_audio/<filename>', methods=['GET'])
def download_audio(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error serving audio file: {e}")
        return jsonify({'error': 'Error retrieving audio file'}), 500


@app.route('/reset', methods=['POST'])
def reset_conversation():
    global conversation_history
    conversation_history.clear()
    return jsonify({'status': 'Conversation reset successfully'})


if __name__ == '__main__':
    app.run()