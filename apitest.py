import os
import google.generativeai as genai
import logging
from textblob import TextBlob  # Import TextBlob for sentiment analysis

# User profile information
user_profile = {
    "name": "Mehmet",
    "age": 15,
    "interests": ["math", "science"],
    "preferred_communication_style": "formal",
    "learning_style": "visual",  # Example of additional preference
    "favorite_subject": "math"   # Another preference
}

# Retrieve and validate API key
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    print("API key is set.")
    genai.configure(api_key=api_key)
else:
    print("API key not set. Please set your GOOGLE_API_KEY environment variable.")
    exit()

# Define logging configuration
log_filename = "chatbot_logs.txt"
MAX_LOG_SIZE = 1 * 1024 * 1024  # 1 MB

def trim_log_file():
    """Trim the oldest entries from the log file if it exceeds the maximum size."""
    with open(log_filename, "r+") as log_file:
        log_file.seek(0, os.SEEK_END)
        file_size = log_file.tell()

        if file_size > MAX_LOG_SIZE:
            log_file.seek(0)
            lines = log_file.readlines()

            size_to_trim = file_size - MAX_LOG_SIZE
            trimmed_lines = []
            current_size = 0
            for line in lines:
                if current_size + len(line) > size_to_trim:
                    trimmed_lines.append(line)
                else:
                    current_size += len(line)

            log_file.seek(0)
            log_file.writelines(trimmed_lines)
            log_file.truncate()

# Configure logging settings
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Define the generative model
model = genai.GenerativeModel("gemini-1.5-flash")

# Define a generic long-term memory with educational themes
long_term_memory = {
    "math": "Mathematics involves problem-solving skills. Key areas include arithmetic, geometry, and algebra.",
    "science": "Science explores natural phenomena. It includes biology, chemistry, physics, and earth sciences.",
    "literature": "Literature is about stories, poetry, and plays. It helps develop imagination and empathy.",
    "history": "History teaches us about past events and important figures. It helps us understand the world today.",
    "geography": "Geography involves learning about places, cultures, and environments around the world."
}

def analyze_sentiment(text):
    """Analyze sentiment of the given text and return the sentiment score."""
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Returns a float within the range [-1.0, 1.0]

def generate_response(prompt, user_profile, session_memory, long_term_memory, sentiment_score):
    """
    Generate a response based on user input, user profile, session memory, long-term memory, and sentiment score.
    
    Args:
        prompt (str): User input prompt.
        user_profile (dict): Information about the user.
        session_memory (list): Ongoing conversation context.
        long_term_memory (dict): General knowledge on various subjects.
        sentiment_score (float): Sentiment score of the user input.
    
    Returns:
        str: Generated response from the model.
    """
    # Construct a personalized context based on user profile
    profile_context = (
        f"This user is named {user_profile['name']}, aged {user_profile['age']}. "
        f"They prefer a {user_profile['preferred_communication_style']} communication style. "
        f"They are interested in {', '.join(user_profile['interests'])}. "
        f"Their favorite subject is {user_profile['favorite_subject']} and they learn best visually."
    )

    # Prepare sentiment context
    sentiment_context = "The user's sentiment score is {:.2f}.".format(sentiment_score)

    # Combine long-term memory themes and profile context with session memory
    relevant_context = " ".join(long_term_memory.values())
    session_context = "\n".join(session_memory[-5:])  # Limit to last 5 interactions for context
    full_prompt = f"{profile_context}\n{sentiment_context}\n{relevant_context}\n{session_context}\nYou: {prompt}\nBot:"

    # Generate response using the full prompt with context
    response = model.generate_content(full_prompt)
    return response

def chatbot_loop():
    """Run the chatbot interaction loop with session memory, generic long-term memory, and sentiment analysis."""
    session_memory = []  # Initialize session memory as a list
    
    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("Chatbot session ended.")
            break

        try:
            # Analyze user sentiment
            sentiment_score = analyze_sentiment(user_input)

            # Generate response using user input, user profile, session memory, long-term memory, and sentiment score
            response = generate_response(user_input, user_profile, session_memory, long_term_memory, sentiment_score)
            
            if response and hasattr(response, 'text'):
                bot_response = response.text
            else:
                bot_response = "Sorry, I couldn't generate a response at this time."

            # Display bot response
            print("Bot:", bot_response)

            # Trim the log file if necessary before logging
            trim_log_file()

            # Log user input and bot response
            logging.info(f"User: {user_input}")
            logging.info(f"Bot: {bot_response}")

            # Update session memory with current interaction
            session_memory.append(f"You: {user_input}\nBot: {bot_response}")  # Append to session memory

        except Exception as e:
            print("Error generating response:", e)

# Start the chatbot loop
if __name__ == "__main__":
    chatbot_loop()
