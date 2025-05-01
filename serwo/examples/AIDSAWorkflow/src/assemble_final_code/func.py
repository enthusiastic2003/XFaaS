# src/assemble_final_code/func.py
import time
import json
from python.src.utils.classes.commons.serwo_objects import SerWOObject
# src/_shared_utils/cost_calculator.py
# Define arbitrary cost constants (adjust as needed)
COST_PER_GB_SEC = 0.00001667
COST_PER_SEC = 0.000001
COST_PER_INVOCATION = 0.0000004

def calculate_cost(memory_mb, duration_sec, llm_token_cost=0):
    """Calculates a hypothetical cost based on memory and duration."""
    if duration_sec < 0: duration_sec = 0
    if memory_mb <= 0: memory_mb = 128

    memory_gb = memory_mb / 1024.0
    compute_cost = (memory_gb * duration_sec * COST_PER_GB_SEC) + \
                   (duration_sec * COST_PER_SEC) + \
                   COST_PER_INVOCATION
    return compute_cost + llm_token_cost


# src/_shared_utils/groq_helper.py
import os
import time
from groq import Groq, APIError, RateLimitError

# Placeholder costs per token - replace with actual values if available from Groq/Ollama
# Using placeholders as Groq pricing might differ or not be public per token easily
COST_PER_MILLION_INPUT = {
    "llama3-70b-8192": 0.00, # Example: Groq often has free tiers or different pricing
    "llama3-8b-8192": 0.00,
    "llama-guard-3-8b": 0.00,
    "llama-3.1-8b-instant": 0.00,
}
COST_PER_MILLION_OUTPUT = {
    "llama3-70b-8192": 0.00,
    "llama3-8b-8192": 0.00,
    "llama-guard-3-8b": 0.00,
    "llama-3.1-8b-instant": 0.00,
}

# Model mapping using aliases
MODEL_MAP = {
    "L1": {"name": "llama3-70b-8192"}, # Divider, Selection, Assembly (Groq)
    "G1": {"name": "llama3-8b-8192"},  # Prompt Gen, Code Gen Model 1 (Groq)
    "G2": {"name": "llama-guard-3-8b"}, # Code Gen Model 2 (Groq - assuming available)
    "G3": {"name": "llama-3.1-8b-instant"},# Code Gen Model 3 (Groq - assuming available)
}
# Note: Ensure models G2 and G3 are actually available via your Groq API key/endpoint.
# If not, you might need to substitute with available models like mixtral-8x7b-32768 or gemma-7b-it

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def call_groq(messages, model_alias, temperature=0.1, max_tokens=4096):
    """
    Calls the Groq API with the specified model alias and messages.
    Returns {"response": content / None, "error": msg / None}, duration, prompt_tokens, completion_tokens.
    """
    model_info = MODEL_MAP.get(model_alias)
    if not model_info:
        return {"error": f"Unknown model alias: {model_alias}"}, 0, 0, 0

    model_name = model_info["name"]

    duration = -1
    response_content = None
    error_msg = None
    prompt_tokens = 0
    completion_tokens = 0
    start_time = None

    if not client.api_key:
        return {"error": "GROQ_API_KEY environment variable not set."}, 0, 0, 0

    try:
        start_time = time.time()
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        end_time = time.time()
        duration = end_time - start_time

        if chat_completion.choices:
            response_content = chat_completion.choices[0].message.content
        else:
            error_msg = "API returned success but no response choices."

        if hasattr(chat_completion, 'usage'): # Check if usage info exists
             prompt_tokens = chat_completion.usage.prompt_tokens
             completion_tokens = chat_completion.usage.completion_tokens

        print(f"Groq call {model_alias}({model_name}) OK: {duration:.3f}s, Tokens(P:{prompt_tokens}, C:{completion_tokens})")

    except RateLimitError as e:
        error_msg = f"API Rate Limit Error ({model_alias}): {str(e)}"
    except APIError as e:
        error_msg = f"API Error ({model_alias}, {e.status_code}): {str(e)}"
    except Exception as e:
        error_msg = f"Unexpected Groq client error ({model_alias}): {str(e)}"

    # Capture duration even on error
    if duration < 0 and start_time:
         duration = time.time() - start_time
    elif duration < 0:
         duration = 0 # Assign 0 if start_time wasn't even set

    if error_msg:
         print(f"Groq call {model_alias}({model_name}) FAILED: {duration:.3f}s, Error: {error_msg}")


    result_data = {"response_content": response_content}
    if error_msg:
        result_data["error"] = error_msg

    return result_data, duration, prompt_tokens, completion_tokens


def calculate_groq_cost(model_alias, prompt_tokens, completion_tokens):
    """ Calculates the estimated cost of a Groq API call based on token usage. """
    model_info = MODEL_MAP.get(model_alias)
    if not model_info: return 0
    model_name = model_info["name"]

    input_cost_per_million = COST_PER_MILLION_INPUT.get(model_name, 0)
    output_cost_per_million = COST_PER_MILLION_OUTPUT.get(model_name, 0)

    input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million
    output_cost = (completion_tokens / 1_000_000) * output_cost_per_million
    return input_cost + output_cost
    
    

NODE_MEMORY_MB = 768
ASSEMBLY_MODEL_ALIAS = "L1"

def function(serwoObject) -> SerWOObject:
    node_start_time = time.time()
    body = serwoObject.get_body()
    metadata = serwoObject.get_metadata()

    problem_description = body.get("problem_description", "No original problem description found.")
    selected_codes_dict = body.get("selected_codes", {})
    node_metrics = body.get("node_metrics", [])
    workflow_path = body.get("workflow_path", [])
    total_duration_sec = body.get("total_duration_sec", 0.0)
    total_cost = body.get("total_cost", 0.0)

    final_assembled_code = None
    assembly_status = "assembly_failed"
    assembly_error = None
    llm_duration = 0
    llm_token_cost = 0

    # Prepare input for assembly prompt
    assembly_input_parts = []
    all_subproblems_valid = True
    for i in range(1, 4):
        key = f"subproblem_{i}"
        data = selected_codes_dict.get(key)
        if data and data.get("code") and data["code"] != "INCORRECT":
            assembly_input_parts.append(f"--- Subproblem {i} Description ---\n{data.get('description', 'N/A')}\n\n--- Selected Code for Subproblem {i} ---\n```python\n{data['code']}\n```")
        else:
            assembly_input_parts.append(f"--- Subproblem {i} Description ---\n{data.get('description', 'N/A') if data else 'N/A'}\n\n--- Selected Code for Subproblem {i} ---\n[CODE WAS INCORRECT OR MISSING]")
            all_subproblems_valid = False # Mark if any part failed

    combined_sub_code = "\n\n".join(assembly_input_parts)

    if not all_subproblems_valid:
         assembly_error = "Cannot assemble final code because one or more subproblem codes were INCORRECT or missing."
         print(f"AssembleFinalCode: {assembly_error}")
    else:
        print("AssembleFinalCode: Assembling final code...")
        assembly_prompt = f"""You are an expert Python programmer assembling a final solution.
Original Problem:
{problem_description}

Subproblem Descriptions and Selected Code Snippets:
{combined_sub_code}

Your Task: Combine the selected code snippets into a single, complete, and correct Python program that solves the original problem.
Ensure necessary imports are included, functions are called correctly, and the overall logic matches the original problem description.
Respond ONLY with the final Python code block, starting with ```python and ending with ```. Do not add explanations outside the code block.
"""
        messages = [{"role": "system", "content": "Assemble Python code from parts. Respond ONLY with the final code block."}, {"role": "user", "content": assembly_prompt}]

        response_data, duration, p_tok, c_tok = call_groq(messages, ASSEMBLY_MODEL_ALIAS, temperature=0.2, max_tokens=4096) # Allow larger output
        llm_duration = duration
        llm_token_cost = calculate_groq_cost(ASSEMBLY_MODEL_ALIAS, p_tok, c_tok)

        if response_data.get("response_content"):
            code = response_data["response_content"].strip().lstrip("```python").rstrip("```").strip()
            if code:
                final_assembled_code = code
                assembly_status = "assembly_success"
                print("AssembleFinalCode: Assembly successful.")
            else:
                assembly_error = "Assembly LLM returned empty code block."
                assembly_status = "assembly_empty_response"
        else:
            assembly_error = f"Assembly LLM failed: {response_data.get('error', 'No response')}"
            assembly_status = "assembly_api_error"

        if assembly_error:
             print(f"AssembleFinalCode: {assembly_error}")

    # Calculate Node Metrics
    node_end_time = time.time()
    node_duration_sec = node_end_time - node_start_time
    node_compute_duration = max(0, node_duration_sec - llm_duration)
    node_cost = calculate_cost(NODE_MEMORY_MB, node_compute_duration, llm_token_cost=llm_token_cost)

    # Update workflow state
    body["final_assembled_code"] = final_assembled_code
    body["assembly_status"] = assembly_status
    body["assembly_error"] = assembly_error

    body["total_duration_sec"] = total_duration_sec + node_duration_sec
    body["total_cost"] = total_cost + node_cost
    node_metrics_details = {
        "node": "AssembleFinalCode", "duration_sec": node_duration_sec, "compute_duration_sec": node_compute_duration,
        "cost": node_cost, "memory_mb": NODE_MEMORY_MB, "status": assembly_status,
        "llm_duration_sec": llm_duration, "llm_token_cost": llm_token_cost,
        "error": assembly_error
    }
    node_metrics.append(node_metrics_details)
    workflow_path.append(f"AssembleFinalCode ({assembly_status})")

    body["node_metrics"] = node_metrics
    body["workflow_path"] = workflow_path
    # Set next node
    body["next_node_name"] = "FinalizeOutput"

    print(f"AssembleFinalCode: Finished. Status='{assembly_status}'. Node Duration={node_duration_sec:.3f}s, Node Cost=${node_cost:.6f}")

    return SerWOObject(body=body, metadata=metadata)
