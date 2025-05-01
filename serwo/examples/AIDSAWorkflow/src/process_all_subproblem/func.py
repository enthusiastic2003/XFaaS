# src/process_all_subproblems/func.py
import time
import json
import re
from python.src.utils.classes.commons.serwo_objects import SerWOObject
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

NODE_MEMORY_MB = 1024 # Assume high memory due to multiple calls / state

# Model Aliases used within this node
DIVIDER_MODEL_ALIAS = "L1"
PROMPT_GEN_MODEL_ALIAS = "G1"
CODE_GEN_MODEL_ALIASES = ["G1", "G2", "G3"] # Always use these 3
SELECTION_MODEL_ALIAS = "L1"

# Separators (Ensure these EXACT strings are used in prompts)
SP_SEP_1_START = "---SUBPROBLEM_1_START---"
SP_SEP_1_END_2_START = "---SUBPROBLEM_1_END------SUBPROBLEM_2_START---"
SP_SEP_2_END_3_START = "---SUBPROBLEM_2_END------SUBPROBLEM_3_START---"
SP_SEP_3_END = "---SUBPROBLEM_3_END---"
CANDIDATE_CODE_SEPARATOR = "---CANDIDATE_CODE_END---"

FALLBACK_SUBPROBLEMS_STRING = f"""{SP_SEP_1_START}
Helper function to convert numbers to binary strings of fixed length.{SP_SEP_1_END_2_START}
Build a Trie (prefix tree) from the binary representations of the numbers.{SP_SEP_2_END_3_START}
For each number, query the Trie to find the number yielding the maximum XOR.{SP_SEP_3_END}"""

def extract_subproblem(subproblems_string, index):
    # (Same regex extraction function as before)
    pattern = None
    if index == 1: pattern = rf"{re.escape(SP_SEP_1_START)}(.*?){re.escape(SP_SEP_1_END_2_START)}"
    elif index == 2: pattern = rf"{re.escape(SP_SEP_1_END_2_START)}(.*?){re.escape(SP_SEP_2_END_3_START)}"
    elif index == 3: pattern = rf"{re.escape(SP_SEP_2_END_3_START)}(.*?){re.escape(SP_SEP_3_END)}"
    else: return None
    match = re.search(pattern, subproblems_string, re.DOTALL)
    return match.group(1).strip() if match else None

def function(serwoObject) -> SerWOObject:
    node_start_time = time.time()
    body = serwoObject.get_body()
    metadata = serwoObject.get_metadata()

    # Extract initial state
    problem_description = body.get("problem_description", "No problem description provided.")
    node_metrics = body.get("node_metrics", [])
    workflow_path = body.get("workflow_path", [])
    total_duration_sec = body.get("total_duration_sec", 0.0)
    total_cost = body.get("total_cost", 0.0)

    node_status = "failed"
    node_error = None
    selected_codes_dict = {}
    subproblem_descriptions_list = []
    llm_calls_metrics = [] # Track LLM calls in this node

    # --- 1. Divide the problem ---
    print("ProcessAllSubproblems: Step 1 - Dividing problem...")
    # (Prompt for division is same as before, using L1)
    division_prompt = f"""You are an expert at breaking down programming problems... (same as before, use the exact separators)

    Problem: {problem_description}

    Your Output (Use EXACT Separators):
    """
    messages = [{"role": "system", "content": f"Divide problem into 3 parts, format strictly using separators like {SP_SEP_1_START}...{SP_SEP_3_END}"}, {"role": "user", "content": division_prompt}]
    response_data, duration, p_tok, c_tok = call_groq(messages, DIVIDER_MODEL_ALIAS, temperature=0.1)
    llm_calls_metrics.append({"step": "divide", "model": DIVIDER_MODEL_ALIAS, "duration": duration, "p_tok": p_tok, "c_tok": c_tok, "error": response_data.get("error")})

    # (Subproblem string extraction and fallback logic same as before)
    subproblems_string = None
    if response_data.get("response_content"):
        llm_output = response_data["response_content"].strip()
        # Basic check for separators
        if all(sep in llm_output for sep in [SP_SEP_1_START, SP_SEP_1_END_2_START, SP_SEP_2_END_3_START, SP_SEP_3_END]):
            subproblems_string = llm_output
            print("ProcessAllSubproblems: Division successful.")
        else: node_error = f"Division LLM incorrect format: {llm_output[:100]}..."
    elif response_data.get("error"): node_error = f"Division LLM Error: {response_data['error']}"
    if not subproblems_string:
        print("ProcessAllSubproblems: Using fallback subproblems.")
        subproblems_string = FALLBACK_SUBPROBLEMS_STRING
        node_error = node_error or "Using fallback for subproblems."

    temp_descriptions = []
    for i in range(1, 4):
        desc = extract_subproblem(subproblems_string, i)
        temp_descriptions.append(desc if desc else f"Fallback description for subproblem {i}")
    subproblem_descriptions_list = temp_descriptions

    # --- 2. & 3. Generate Codes and Select for Each Subproblem ---
    print("ProcessAllSubproblems: Step 2/3 - Generating & Selecting codes...")
    generation_selection_successful_overall = True

    for i in range(3): # Loop through subproblems 0, 1, 2
        subproblem_index = i + 1
        current_subproblem_desc = subproblem_descriptions_list[i]
        print(f"\nProcessAllSubproblems: --- Subproblem {subproblem_index} ---")

        # --- 2a. Generate Optimized Prompt ---
        print(f"  Generating optimized prompt...")
        # (Prompt for prompt generation same as before, using G1)
        prompt_gen_prompt = f"Generate an optimized and detailed prompt for a Python code generation LLM based on this subproblem description: {current_subproblem_desc}..." # Same as before
        messages = [{"role": "system", "content": "Generate optimized prompt for code gen."}, {"role": "user", "content": prompt_gen_prompt}]
        response_data, duration, p_tok, c_tok = call_groq(messages, PROMPT_GEN_MODEL_ALIAS, temperature=0.5)
        llm_calls_metrics.append({"step": f"prompt_gen_{subproblem_index}", "model": PROMPT_GEN_MODEL_ALIAS, "duration": duration, "p_tok": p_tok, "c_tok": c_tok, "error": response_data.get("error")})

        optimized_prompt = current_subproblem_desc # Fallback
        if response_data.get("response_content"):
            optimized_prompt = response_data["response_content"].strip()
        else:
             node_error = node_error or f"Prompt Gen failed for Subproblem {subproblem_index}"
             generation_selection_successful_overall = False # Mark as issue

        # --- 2c. Call Code Generation Models (Always 3) ---
        generated_code_parts = []
        print(f"  Generating code using {CODE_GEN_MODEL_ALIASES}...")
        for alias in CODE_GEN_MODEL_ALIASES:
            # (Prompt for code gen same as before, using optimized_prompt)
            messages = [{"role": "system", "content": "You are an expert Python programmer. Respond ONLY with the code block (```python...```)."}, {"role": "user", "content": optimized_prompt}]
            response_data, duration, p_tok, c_tok = call_groq(messages, model_alias=alias, temperature=0.7)
            llm_calls_metrics.append({"step": f"code_gen_{subproblem_index}_{alias}", "model": alias, "duration": duration, "p_tok": p_tok, "c_tok": c_tok, "error": response_data.get("error")})

            # (Code extraction logic same as before)
            generated_code_part = f"---ERROR_FROM_MODEL_{alias}---{response_data.get('error', 'No response')}"
            if response_data.get("response_content"):
                 code = response_data["response_content"].strip().lstrip("```python").rstrip("```").strip()
                 generated_code_part = code if code else f"---EMPTY_RESPONSE_FROM_MODEL_{alias}---"
            else:
                 node_error = node_error or f"Code Gen {alias} failed for Subproblem {subproblem_index}"
                 generation_selection_successful_overall = False
            generated_code_parts.append(generated_code_part)

        combined_generated_code_string = CANDIDATE_CODE_SEPARATOR.join(generated_code_parts)

        # --- 3. Select Code ---
        print(f"  Selecting best code...")
        # (Prompt for selection same as before, using combined codes and description, using L1)
        selection_prompt = f"""You are an expert code reviewer... (same as before)

        Code Snippets:{combined_generated_code_string}

        Your Task: Select the best code for Subproblem {subproblem_index} ("{current_subproblem_desc}")..."""
        messages = [{"role": "system", "content": "Select the best Python code or respond 'INCORRECT'. Respond ONLY with the code block or 'INCORRECT'."}, {"role": "user", "content": selection_prompt}]
        response_data, duration, p_tok, c_tok = call_groq(messages, SELECTION_MODEL_ALIAS, temperature=0.1)
        llm_calls_metrics.append({"step": f"select_{subproblem_index}", "model": SELECTION_MODEL_ALIAS, "duration": duration, "p_tok": p_tok, "c_tok": c_tok, "error": response_data.get("error")})

        # (Selection logic same as before)
        selected_code = None
        selection_status = "failed_to_select"
        selection_error = None
        if response_data.get("response_content"):
            llm_output = response_data["response_content"].strip()
            if llm_output == "INCORRECT":
                selected_code = "INCORRECT"; selection_status = "selected_incorrect"
            else:
                code = llm_output.lstrip("```python").rstrip("```").strip()
                if code: selected_code = code; selection_status = "selected"
                else: selection_error = "Selection LLM returned empty block"; selection_status="selection_empty_response"
        else: selection_error = f"Selection LLM failed: {response_data.get('error', 'No response')}"

        if selection_error:
             node_error = node_error or selection_error
             generation_selection_successful_overall = False

        selected_codes_dict[f"subproblem_{subproblem_index}"] = {
            "code": selected_code, "status": selection_status, "error": selection_error, "description": current_subproblem_desc
        }
        print(f"  Selection Status: {selection_status}")


    # --- Determine overall node status ---
    if generation_selection_successful_overall and not node_error:
        node_status = "completed_success"
    else:
        node_status = "completed_with_issues"


    # --- Calculate Node Metrics ---
    node_end_time = time.time()
    node_duration_sec = node_end_time - node_start_time
    # (Calculate total_llm_duration, total_llm_token_cost, node_compute_duration, node_cost same as before)
    total_llm_duration = sum(m['duration'] for m in llm_calls_metrics if m['duration'] >= 0)
    total_llm_token_cost = sum(calculate_groq_cost(m['model'], m['p_tok'], m['c_tok']) for m in llm_calls_metrics)
    total_llm_api_errors = sum(1 for m in llm_calls_metrics if m['error'])
    node_compute_duration = max(0, node_duration_sec - total_llm_duration)
    node_cost = calculate_cost(NODE_MEMORY_MB, node_compute_duration, llm_token_cost=total_llm_token_cost)


    # Update workflow state
    body["selected_codes"] = selected_codes_dict
    body["subproblem_descriptions_list"] = subproblem_descriptions_list

    body["total_duration_sec"] = total_duration_sec + node_duration_sec
    body["total_cost"] = total_cost + node_cost
    node_metrics_details = {
        "node": "ProcessAllSubproblems", "duration_sec": node_duration_sec, "compute_duration_sec": node_compute_duration,
        "cost": node_cost, "memory_mb": NODE_MEMORY_MB, "status": node_status,
        "total_llm_duration_sec": total_llm_duration, "total_llm_token_cost": total_llm_token_cost,
        "total_llm_calls": len(llm_calls_metrics), "total_llm_api_errors": total_llm_api_errors,
        "error": node_error # Aggregate errors potentially
        # "llm_calls_breakdown": llm_calls_metrics # Optional detailed breakdown
    }
    node_metrics.append(node_metrics_details)
    workflow_path.append(f"ProcessAllSubproblems ({node_status})")

    body["node_metrics"] = node_metrics
    body["workflow_path"] = workflow_path
    # Set next node
    body["next_node_name"] = "AssembleFinalCode"

    print(f"ProcessAllSubproblems: Finished. Status='{node_status}'. Node Duration={node_duration_sec:.3f}s, Node Cost=${node_cost:.6f}")
    print(f"  Total Workflow Duration={body['total_duration_sec']:.3f}s, Total Workflow Cost=${body['total_cost']:.6f}")

    return SerWOObject(body=body, metadata=metadata)
