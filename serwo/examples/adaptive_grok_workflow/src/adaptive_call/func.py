import os
import time
from groq import Groq
from python.src.utils.classes.commons.serwo_objects import SerWOObject

# --- Configuration ---
API_KEY = "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"  # Ensure this environment variable is set securely
MODEL = "llama3-70b-8192"  # Adjust to your preferred Groq-supported model
LATENCY_THRESHOLD_SEC = 6.0  # Adjust this threshold as needed

# --- Initialize Groq Client ---
client = Groq(api_key=API_KEY)

# --- Helper Function ---
def call_groq(messages, temperature=0.5):
    """
    Calls the Groq API and returns the response and duration.
    """
    duration = -1
    groq_response = None
    error_msg = None

    if not API_KEY:
        return {"error": "GROQ_API_KEY environment variable not set."}, duration

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            messages=messages,
            model=MODEL,
            temperature=temperature,
            stream=False
        )
        end_time = time.time()
        duration = end_time - start_time
        groq_response = response
    except Exception as e:
        error_msg = f"API request failed: {str(e)}"
        if 'start_time' in locals():
            duration = time.time() - start_time

    result_data = {"response": groq_response}
    if error_msg:
        result_data["error"] = error_msg

    return result_data, duration
# --- End Helper ---

def function(serwoObject) -> SerWOObject:
    metadata = serwoObject.get_metadata()
    input_body = serwoObject.get_body()

    initial_topic = input_body.get("initial_topic", "Unknown topic")
    initial_response_data = input_body.get("initial_call_response_data", {})
    initial_duration = input_body.get("initial_call_duration_sec", -1)
    initial_call_error = initial_response_data.get("error")
    initial_call_content = None

    if not initial_call_error and initial_response_data.get("response"):
        try:
            initial_call_content = initial_response_data["response"].choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            initial_call_content = "Error extracting content from initial response."

    # --- Adapt prompt based on latency ---
    adapted_prompt = ""
    strategy_used = ""
    temperature_used = 0.5  # Default

    if initial_duration > 0 and initial_duration < LATENCY_THRESHOLD_SEC and not initial_call_error:
        strategy_used = f"below_threshold ({LATENCY_THRESHOLD_SEC}s)"
        adapted_prompt = (
            f"Regarding '{initial_topic}', you previously mentioned (summary): "
            f"'{str(initial_call_content)[:200]}...'. Now, please elaborate specifically on the potential long-term "
            f"economic consequences mentioned or implied in the initial discussion. Be specific."
        )
        temperature_used = 0.6  # Allow more elaboration
    else:
        strategy_used = f"above_threshold ({LATENCY_THRESHOLD_SEC}s) or error"
        if initial_call_content and not initial_call_error:
            adapted_prompt = (
                f"Regarding '{initial_topic}', considering the previous information: "
                f"'{str(initial_call_content)[:200]}...'. Provide only a 3-bullet point summary of the absolute key takeaways."
            )
        else:
            adapted_prompt = f"Provide a very brief, 2-sentence definition of '{initial_topic}'."
        temperature_used = 0.2  # Be concise

    print(f"AdaptiveCall: Initial Latency={initial_duration:.4f}s. Strategy='{strategy_used}'")
    print(f"AdaptiveCall: Using adapted prompt: {adapted_prompt}")
    # ---

    messages = [
        {"role": "system", "content": "You are responding to a follow-up request based on previous interaction."},
        {"role": "user", "content": adapted_prompt}
    ]

    response_data_2, duration_2 = call_groq(messages, temperature=temperature_used)

    # --- Prepare Final Output Body ---
    final_output_body = {
        "initial_topic": initial_topic,
        "initial_call_latency_sec": initial_duration,
        "latency_threshold_sec": LATENCY_THRESHOLD_SEC,
        "adaptation_strategy_used": strategy_used,
        "adapted_prompt_used": adapted_prompt,
        "final_call_latency_sec": duration_2,
        "final_call_response_data": response_data_2,
        "final_answer": None
    }

    final_error = response_data_2.get("error")
    if not final_error and response_data_2.get("response"):
        try:
            final_output_body["final_answer"] = response_data_2["response"].choices[0].message.content
        except (AttributeError, IndexError, TypeError) as e:
            final_output_body["final_answer"] = f"Error extracting final answer content: {str(e)}"
            final_output_body["final_call_response_data"]["error"] = final_output_body["final_answer"]
    elif final_error:
        final_output_body["final_answer"] = f"Final API call failed: {final_error}"
    else:
        final_output_body["final_answer"] = "Final API call did not return a valid response structure."

    print(f"AdaptiveCall: Final Duration={duration_2:.4f}s, Final Answer Snippet='{str(final_output_body['final_answer'])[:100]}...'")

    return SerWOObject(body=final_output_body, metadata=metadata)
