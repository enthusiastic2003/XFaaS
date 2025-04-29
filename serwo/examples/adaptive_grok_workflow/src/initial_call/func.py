import os
import time
from python.src.utils.classes.commons.serwo_objects import SerWOObject
from groq import Groq

# Initialize the Groq client with your API key
client = Groq(api_key="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")
MODEL = "llama3-8b-8192"  # Replace with your desired model

# --- Helper Function ---
def call_groq(messages):
    """Calls the Groq API and returns the response and duration."""
    duration = -1
    groq_response = None
    error_msg = None

    if not client.api_key:
        return {"error": "GROQ_API_KEY environment variable not set."}, duration

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            messages=messages,
            model=MODEL,
            temperature=0.5,
            stream=False
        )
        end_time = time.time()
        duration = end_time - start_time
        groq_response = response.choices[0].message.content
    except Exception as e:
        error_msg = f"API request failed: {str(e)}"
        if start_time:
            duration = time.time() - start_time

    if duration < 0 and 'start_time' in locals():
        duration = time.time() - start_time

    result_data = {"response": groq_response}
    if error_msg:
        result_data["error"] = error_msg

    return result_data, duration
# --- End Helper ---

def function(serwoObject) -> SerWOObject:
    metadata = serwoObject.get_metadata()
    body = serwoObject.get_body()

    initial_topic = body.get("initial_topic", "The impact of AI on software development productivity.")
    initial_prompt = f"Provide a balanced overview of {initial_topic}. Discuss both potential benefits (like code generation, bug detection) and challenges (like job displacement, new skill requirements, ethical concerns). Aim for a comprehensive yet readable summary."

    print(f"InitialCall: Starting call for topic: {initial_topic}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant providing balanced information."},
        {"role": "user", "content": initial_prompt}
    ]

    response_data, duration = call_groq(messages)

    output_body = {
        "initial_topic": initial_topic,
        "initial_prompt_used": initial_prompt,
        "initial_call_response_data": response_data,
        "initial_call_duration_sec": duration
    }
    print(f"InitialCall: Duration={duration:.4f}s, Error='{response_data.get('error')}'")

    return SerWOObject(body=output_body, metadata=metadata)
