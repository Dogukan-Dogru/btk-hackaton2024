import os
import google.generativeai as genai
import logging
from textblob import TextBlob  

user_profile = {
    "name": "Mehmet",
    "age": 15,
    "interests": ["math", "science"],
    "preferred_communication_style": "formal",
    "learning_style": "visual",  
    "favorite_subject": "math"   
}

api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    print("API key is set.")
    genai.configure(api_key=api_key)
else:
    print("API key not set. Please set your GOOGLE_API_KEY environment variable.")
    exit()

log_filename = "chatbot_logs.txt"
MAX_LOG_SIZE = 1 * 1024 * 1024 

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

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

model = genai.GenerativeModel("gemini-1.5-flash")

long_term_memory = {
    "math": "Mathematics involves problem-solving skills. Key areas include arithmetic, geometry, and algebra.",
    "science": "Science explores natural phenomena. It includes biology, chemistry, physics, and earth sciences.",
    "literature": "Literature is about stories, poetry, and plays. It helps develop imagination and empathy.",
    "history": "History teaches us about past events and important figures. It helps us understand the world today.",
    "geography": "Geography involves learning about places, cultures, and environments around the world."
}

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  

def generate_response(prompt, user_profile, session_memory, long_term_memory, sentiment_score):

    profile_context = (
        f"This user is named {user_profile['name']}, aged {user_profile['age']}. "
        f"They prefer a {user_profile['preferred_communication_style']} communication style. "
        f"They are interested in {', '.join(user_profile['interests'])}. "
        f"Their favorite subject is {user_profile['favorite_subject']} and they learn best visually."
    )

    sentiment_context = "The user's sentiment score is {:.2f}.".format(sentiment_score)

    relevant_context = " ".join(long_term_memory.values())
    session_context = "\n".join(session_memory[-5:])  
    full_prompt = f"{profile_context}\n{sentiment_context}\n{relevant_context}\n{session_context}\nYou: {prompt}\nBot:"

    response = model.generate_content(full_prompt)
    return response

def chatbot_loop():
    """Run the chatbot interaction loop with session memory, generic long-term memory, and sentiment analysis."""
    session_memory = [] 
    
    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("Chatbot session ended.")
            break

        try:
            sentiment_score = analyze_sentiment(user_input)

            response = generate_response(user_input, user_profile, session_memory, long_term_memory, sentiment_score)
            
            if response and hasattr(response, 'text'):
                bot_response = response.text
            else:
                bot_response = "err"

            print("Bot:", bot_response)

            trim_log_file()

            logging.info(f"User: {user_input}")
            logging.info(f"Bot: {bot_response}")

            session_memory.append(f"You: {user_input}\nBot: {bot_response}") 

        except Exception as e:
            print("Error generating response:", e)

if __name__ == "__main__":
    chatbot_loop()
