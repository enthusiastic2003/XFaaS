import time
import json
import logging
from python.src.utils.classes.commons.serwo_objects import SerWOObject
from groq import Groq, APIError, RateLimitError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Groq client
client = Groq(api_key="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")

# Constants
NODE_MEMORY_MB = 768
ASSEMBLY_MODEL_ALIAS = "L1"
COST_PER_GB_SEC = 0.00001667
COST_PER_SEC = 0.000001
COST_PER_INVOCATION = 0.0000004

# Model mapping
MODEL_MAP = {
    "L1": {"name": "llama3-70b-8192"},  # Assembly model
    "G1": {"name": "llama3-8b-8192"},
    "G2": {"name": "llama-guard-3-8b"},
    "G3": {"name": "llama-3.1-8b-instant"},
}


def calculate_cost(memory_mb, duration_sec, llm_token_cost=0):
    """Calculate compute cost based on memory usage and duration"""
    memory_gb = max(128, memory_mb) / 1024.0
    duration_sec = max(0, duration_sec)
    compute_cost = (memory_gb * duration_sec * COST_PER_GB_SEC) + \
                  (duration_sec * COST_PER_SEC) + \
                  COST_PER_INVOCATION
    return compute_cost + llm_token_cost


def call_groq(messages, model_alias, temperature=0.1, max_tokens=4096):
    """Call Groq API with specified model and parameters"""
    model_name = MODEL_MAP.get(model_alias, {}).get("name")
    if not model_name:
        return {"error": f"Unknown model alias: {model_alias}"}, 0, 0, 0
    
    start_time = time.time()
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        duration = time.time() - start_time
        
        # Extract content and token usage
        response_content = chat_completion.choices[0].message.content if chat_completion.choices else None
        prompt_tokens = getattr(chat_completion.usage, 'prompt_tokens', 0)
        completion_tokens = getattr(chat_completion.usage, 'completion_tokens', 0)
        
        logger.info(f"Groq call {model_alias}({model_name}) OK: {duration:.3f}s, Tokens(P:{prompt_tokens}, C:{completion_tokens})")
        return {"response_content": response_content}, duration, prompt_tokens, completion_tokens
        
    except (RateLimitError, APIError, Exception) as e:
        duration = time.time() - start_time
        error_msg = f"Groq client error ({model_alias}): {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}, duration, 0, 0


def calculate_groq_cost(model_alias, prompt_tokens, completion_tokens):
    """Calculate estimated cost of a Groq API call"""
    # This is a placeholder since costs are set to 0 in the original code
    return 0


def function(serwoObject) -> SerWOObject:
    try:
        node_start_time = time.time()
        body = serwoObject.get_body()
        metadata = serwoObject.get_metadata()

        # Extract workflow state
        problem_description = body.get("problem_description", "No original problem description found.")
        selected_codes_dict = body.get("selected_codes", {})
        node_metrics = body.get("node_metrics", [])
        workflow_path = body.get("workflow_path", [])
        total_duration_sec = body.get("total_duration_sec", 0.0)
        total_cost = body.get("total_cost", 0.0)

        # Initialize variables
        final_assembled_code = None
        assembly_status = "assembly_failed"
        assembly_error = None
        llm_duration = 0
        llm_token_cost = 0

        # Prepare subproblem code parts for assembly
        assembly_input_parts = []
        all_subproblems_valid = True
        
        for i in range(1, 4):
            key = f"subproblem_{i}"
            data = selected_codes_dict.get(key, {})
            code = data.get("code")
            description = data.get("description", "N/A")
            
            if code and code != "INCORRECT":
                assembly_input_parts.append(
                    f"--- Subproblem {i} Description ---\n{description}\n\n"
                    f"--- Selected Code for Subproblem {i} ---\n```python\n{code}\n```"
                )
            else:
                assembly_input_parts.append(
                    f"--- Subproblem {i} Description ---\n{description}\n\n"
                    f"--- Selected Code for Subproblem {i} ---\n[CODE WAS INCORRECT OR MISSING]"
                )
                all_subproblems_valid = False

        combined_sub_code = "\n\n".join(assembly_input_parts)

        # Assemble final code if all subproblems have valid solutions
        if not all_subproblems_valid:
            assembly_error = "Cannot assemble final code because one or more subproblem codes were INCORRECT or missing."
            logger.warning(assembly_error)
        else:
            logger.info("Assembling final code...")
            assembly_prompt = f"""You are an expert Python programmer assembling a final solution.
Original Problem:
{problem_description}

Subproblem Descriptions and Selected Code Snippets:
{combined_sub_code}

Your Task: Combine the selected code snippets into a single, complete, and correct Python program that solves the original problem.
Ensure necessary imports are included, functions are called correctly, and the overall logic matches the original problem description.
Respond ONLY with the final Python code block, starting with ```python and ending with ```. Do not add explanations outside the code block.
"""
            messages = [
                {"role": "system", "content": "Assemble Python code from parts. Respond ONLY with the final code block."},
                {"role": "user", "content": assembly_prompt}
            ]

            response_data, duration, p_tok, c_tok = call_groq(messages, ASSEMBLY_MODEL_ALIAS, temperature=0.2, max_tokens=4096)
            llm_duration = duration
            llm_token_cost = calculate_groq_cost(ASSEMBLY_MODEL_ALIAS, p_tok, c_tok)

            if response_data.get("response_content"):
                code = response_data["response_content"].strip().lstrip("```python").rstrip("```").strip()
                if code:
                    final_assembled_code = code
                    assembly_status = "assembly_success"
                    logger.info("Assembly successful.")
                else:
                    assembly_error = "Assembly LLM returned empty code block."
                    assembly_status = "assembly_empty_response"
                    logger.error(assembly_error)
            else:
                assembly_error = f"Assembly LLM failed: {response_data.get('error', 'No response')}"
                assembly_status = "assembly_api_error"
                logger.error(assembly_error)

        # Calculate metrics
        node_duration_sec = time.time() - node_start_time
        node_compute_duration = max(0, node_duration_sec - llm_duration)
        node_cost = calculate_cost(NODE_MEMORY_MB, node_compute_duration, llm_token_cost=llm_token_cost)

        # Update workflow state
        body.update({
            "final_assembled_code": final_assembled_code,
            "assembly_status": assembly_status,
            "assembly_error": assembly_error,
            "total_duration_sec": total_duration_sec + node_duration_sec,
            "total_cost": total_cost + node_cost,
            "next_node_name": "FinalizeOutput"
        })
        
        # Add node metrics
        node_metrics.append({
            "node": "AssembleFinalCode",
            "duration_sec": node_duration_sec,
            "compute_duration_sec": node_compute_duration,
            "cost": node_cost,
            "memory_mb": NODE_MEMORY_MB,
            "status": assembly_status,
            "llm_duration_sec": llm_duration,
            "llm_token_cost": llm_token_cost,
            "error": assembly_error
        })
        
        workflow_path.append(f"AssembleFinalCode ({assembly_status})")
        body["node_metrics"] = node_metrics
        body["workflow_path"] = workflow_path
        
        logger.info(f"Finished with status '{assembly_status}'. Duration={node_duration_sec:.3f}s, Cost=${node_cost:.6f}")
        
        return SerWOObject(body=body, metadata=metadata)
        
    except Exception as e:
        logger.exception(f"Unhandled exception in AssembleFinalCode: {str(e)}")
        # Create minimal response with error
        body = serwoObject.get_body() if hasattr(serwoObject, 'get_body') else {}
        metadata = serwoObject.get_metadata() if hasattr(serwoObject, 'get_metadata') else {}
        
        node_metrics = body.get("node_metrics", [])
        node_metrics.append({
            "node": "AssembleFinalCode",
            "status": "failed",
            "error": f"Unhandled exception: {str(e)}"
        })
        body["node_metrics"] = node_metrics
        body["workflow_path"] = body.get("workflow_path", []) + ["AssembleFinalCode (failed)"]
        body["next_node_name"] = "FinalizeOutput"  # Continue workflow even on failure
        
        return SerWOObject(body=body, metadata=metadata)