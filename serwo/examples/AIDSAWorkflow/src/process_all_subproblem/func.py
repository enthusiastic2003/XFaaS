import time
import json
import re
import logging
from python.src.utils.classes.commons.serwo_objects import SerWOObject
from groq import Groq, APIError, RateLimitError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Groq client
client = Groq(api_key="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")

# Model mapping using aliases
MODEL_MAP = {
    "L1": {"name": "llama3-70b-8192"},  # Divider, Selection models
    "G1": {"name": "llama3-8b-8192"},   # Prompt Gen, Code Gen Model 1
    "G2": {"name": "llama-guard-3-8b"}, # Code Gen Model 2
    "G3": {"name": "llama-3.1-8b-instant"}, # Code Gen Model 3
}

# Separators for subproblem extraction
SP_SEP_1_START = "---SUBPROBLEM_1_START---"
SP_SEP_1_END_2_START = "---SUBPROBLEM_1_END------SUBPROBLEM_2_START---"
SP_SEP_2_END_3_START = "---SUBPROBLEM_2_END------SUBPROBLEM_3_START---"
SP_SEP_3_END = "---SUBPROBLEM_3_END---"
CANDIDATE_CODE_SEPARATOR = "---CANDIDATE_CODE_END---"

# Constants for cost calculation
NODE_MEMORY_MB = 1024
COST_PER_GB_SEC = 0.00001667
COST_PER_SEC = 0.000001
COST_PER_INVOCATION = 0.0000004

# Fallback subproblems if division fails
FALLBACK_SUBPROBLEMS_STRING = f"""{SP_SEP_1_START}
Helper function to convert numbers to binary strings of fixed length.{SP_SEP_1_END_2_START}
Build a Trie (prefix tree) from the binary representations of the numbers.{SP_SEP_2_END_3_START}
For each number, query the Trie to find the number yielding the maximum XOR.{SP_SEP_3_END}"""


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
            max_tokens=max_tokens,
            stream=False,
            temperature=0.0
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


def calculate_cost(memory_mb, duration_sec, llm_token_cost=0):
    """Calculate compute cost based on memory usage and duration"""
    memory_gb = max(128, memory_mb) / 1024.0
    duration_sec = max(0, duration_sec)
    compute_cost = (memory_gb * duration_sec * COST_PER_GB_SEC) + \
                  (duration_sec * COST_PER_SEC) + \
                  COST_PER_INVOCATION
    return compute_cost + llm_token_cost


def extract_subproblem(subproblems_string, index):
    """Extract a specific subproblem from the formatted string"""
    pattern_map = {
        1: r"---SUBPROBLEM_1_START---(.*?)---SUBPROBLEM_1_END---",
        2: r"---SUBPROBLEM_2_START---(.*?)---SUBPROBLEM_2_END---",
        3: r"---SUBPROBLEM_3_START---(.*?)---SUBPROBLEM_3_END---",
    }
    pattern = pattern_map.get(index)
    if not pattern:
        return None
    match = re.search(pattern, subproblems_string, re.DOTALL)
    return match.group(1).strip() if match else None


def function(serwoObject) -> SerWOObject:
    try:
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
        llm_calls_metrics = []
        
        # Step 1: Divide the problem into subproblems
        # 1) The SYSTEM prompt: strict, no fluff
        system_prompt = f"""
        You are a problem decomposition engine.  
        You must split the given problem into exactly three subproblems.  

        **Your output must be NOTHING BUT** the following, in this exact order, with no extra whitespace, no extra words, and no commentary:

        ---SUBPROBLEM_1_START---
        <description of subproblem 1>
        ---SUBPROBLEM_1_END---
        ---SUBPROBLEM_2_START---
        <description of subproblem 2>
        ---SUBPROBLEM_2_END---
        ---SUBPROBLEM_3_START---
        <description of subproblem 3>
        ---SUBPROBLEM_3_END---

        If you cannot produce this exact format, output the single word:
        ERROR
        """

        # 2) The USER prompt: feed in the problem
        user_prompt = f"""
        Here is the problem to decompose:

        \"\"\"
        {problem_description}
        \"\"\"

        Remember: output ONLY the markers and the text between them, or ERROR.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]



        
        response_data, duration, p_tok, c_tok = call_groq(messages, "L1", temperature=0.1)
        llm_calls_metrics.append({
            "step": "divide", 
            "model": "L1", 
            "duration": duration, 
            "p_tok": p_tok, 
            "c_tok": c_tok, 
            "error": response_data.get("error")
        })
        
        logger.info(f"Division LLM response: {response_data.get('response_content', 'No response')}")

        # Extract subproblems or use fallback
        subproblems_string = None
        node_error = None
        if response_data.get("response_content"):
            llm_output = response_data["response_content"].strip()
            EXPECTED_SEPARATORS = [
                "---SUBPROBLEM_1_START---",
                "---SUBPROBLEM_1_END---",
                "---SUBPROBLEM_2_START---",
                "---SUBPROBLEM_2_END---",
                "---SUBPROBLEM_3_START---",
                "---SUBPROBLEM_3_END---",
            ]
            if all(sep in llm_output for sep in EXPECTED_SEPARATORS):
                subproblems_string = llm_output
                logger.info("Division successful.")
                node_error = None
            else:
                node_error = "Division LLM incorrect format"
        else:
            node_error = f"Division LLM Error: {response_data.get('error')}"
        
        if node_error:
            logger.info(node_error)

        
        if not subproblems_string:
            logger.warning("Using fallback subproblems.")
            subproblems_string = FALLBACK_SUBPROBLEMS_STRING
            node_error = node_error or "Using fallback for subproblems."
            
        # Extract subproblem descriptions
        subproblem_descriptions_list = []
        for i in range(1, 4):
            desc = extract_subproblem(subproblems_string, i)
            if not desc:
                logger.warning(f"Subproblem {i} extraction failed.")
                node_error = node_error or f"Subproblem {i} extraction failed."
            subproblem_descriptions_list.append(desc or f"Fallback description for subproblem {i}")
        
        # Step 2 & 3: Generate and select codes for each subproblem
        logger.info("Step 2/3 - Generating & Selecting codes...")
        generation_selection_successful = True
        
        for subproblem_index in range(1, 4):
            current_desc = subproblem_descriptions_list[subproblem_index-1]
            logger.info(f"Processing Subproblem {subproblem_index}")
            
            # Generate optimized prompt
            prompt_gen_prompt = f"Generate an optimized prompt for code generation based on: {current_desc}"
            messages = [
                {"role": "system", "content": "Generate optimized prompt for code gen."},
                {"role": "user", "content": prompt_gen_prompt}
            ]
            response_data, duration, p_tok, c_tok = call_groq(messages, "G1", temperature=0.5)
            llm_calls_metrics.append({
                "step": f"prompt_gen_{subproblem_index}", 
                "model": "G1", 
                "duration": duration, 
                "p_tok": p_tok, 
                "c_tok": c_tok, 
                "error": response_data.get("error")
            })
            
            optimized_prompt = response_data.get("response_content", current_desc).strip()
            if not response_data.get("response_content"):
                node_error = node_error or f"Prompt Gen failed for Subproblem {subproblem_index}"
                generation_selection_successful = False
            
            # Generate code using multiple models
            generated_code_parts = []
            for alias in ["G1", "G2", "G3"]:
                messages = [
                    {"role": "system", "content": "You are an expert Python programmer. Respond ONLY with the code block."},
                    {"role": "user", "content": optimized_prompt}
                ]
                response_data, duration, p_tok, c_tok = call_groq(messages, alias, temperature=0.7)
                llm_calls_metrics.append({
                    "step": f"code_gen_{subproblem_index}_{alias}", 
                    "model": alias, 
                    "duration": duration, 
                    "p_tok": p_tok, 
                    "c_tok": c_tok, 
                    "error": response_data.get("error")
                })
                
                if response_data.get("response_content"):
                    code = response_data["response_content"].strip().lstrip("```python").rstrip("```").strip()
                    generated_code_parts.append(code or f"---EMPTY_RESPONSE_FROM_MODEL_{alias}---")
                else:
                    generated_code_parts.append(f"---ERROR_FROM_MODEL_{alias}---{response_data.get('error', 'No response')}")
                    node_error = node_error or f"Code Gen {alias} failed for Subproblem {subproblem_index}"
                    generation_selection_successful = False
            
            # Select best code
            combined_code = CANDIDATE_CODE_SEPARATOR.join(generated_code_parts)
            selection_prompt = f"Select the best code for Subproblem {subproblem_index} ({current_desc}): {combined_code}"
            messages = [
                {"role": "system", "content": "Select the best Python code or respond 'INCORRECT'."},
                {"role": "user", "content": selection_prompt}
            ]
            response_data, duration, p_tok, c_tok = call_groq(messages, "L1", temperature=0.1)
            llm_calls_metrics.append({
                "step": f"select_{subproblem_index}", 
                "model": "L1", 
                "duration": duration, 
                "p_tok": p_tok, 
                "c_tok": c_tok, 
                "error": response_data.get("error")
            })
            
            # Process selection result
            selection_status = "failed_to_select"
            selection_error = None
            selected_code = None
            
            if response_data.get("response_content"):
                llm_output = response_data["response_content"].strip()
                if llm_output == "INCORRECT":
                    selected_code = "INCORRECT"
                    selection_status = "selected_incorrect"
                else:
                    code = llm_output.lstrip("```python").rstrip("```").strip()
                    if code:
                        selected_code = code
                        selection_status = "selected"
                    else:
                        selection_error = "Selection LLM returned empty block"
                        selection_status = "selection_empty_response"
            else:
                selection_error = f"Selection LLM failed: {response_data.get('error', 'No response')}"
            
            if selection_error:
                node_error = node_error or selection_error
                generation_selection_successful = False
            
            selected_codes_dict[f"subproblem_{subproblem_index}"] = {
                "code": selected_code,
                "status": selection_status,
                "error": selection_error,
                "description": current_desc
            }
            logger.info(f"Selection Status for subproblem {subproblem_index}: {selection_status}")
        
        # Determine overall node status
        node_status = "completed_success" if generation_selection_successful and not node_error else "completed_with_issues"
        
        # Calculate metrics
        node_duration_sec = time.time() - node_start_time
        total_llm_duration = sum(m['duration'] for m in llm_calls_metrics if m['duration'] >= 0)
        total_llm_api_errors = sum(1 for m in llm_calls_metrics if m.get('error'))
        node_compute_duration = max(0, node_duration_sec - total_llm_duration)
        node_cost = calculate_cost(NODE_MEMORY_MB, node_compute_duration)
        
        # Update workflow state
        body.update({
            "selected_codes": selected_codes_dict,
            "subproblem_descriptions_list": subproblem_descriptions_list,
            "total_duration_sec": total_duration_sec + node_duration_sec,
            "total_cost": total_cost + node_cost,
            "next_node_name": "AssembleFinalCode"
        })
        
        # Add node metrics
        node_metrics.append({
            "node": "ProcessAllSubproblems",
            "duration_sec": node_duration_sec,
            "compute_duration_sec": node_compute_duration,
            "cost": node_cost,
            "memory_mb": NODE_MEMORY_MB,
            "status": node_status,
            "total_llm_duration_sec": total_llm_duration,
            "total_llm_calls": len(llm_calls_metrics),
            "total_llm_api_errors": total_llm_api_errors,
            "error": node_error
        })
        
        workflow_path.append(f"ProcessAllSubproblems ({node_status})")
        body["node_metrics"] = node_metrics
        body["workflow_path"] = workflow_path

        logger.info("body: %s", json.dumps(body, indent=2))
        logger.info(f"Finished with status '{node_status}'. Duration={node_duration_sec:.3f}s, Cost=${node_cost:.6f}")
        logger.info(f"Total Workflow Duration={body['total_duration_sec']:.3f}s, Total Workflow Cost=${body['total_cost']:.6f}")
        
        return SerWOObject(body=body, metadata=metadata)
        
    except Exception as e:
        logger.exception(f"Unhandled exception in ProcessAllSubproblems: {str(e)}")
        # Create minimal response with error
        body = serwoObject.get_body() if hasattr(serwoObject, 'get_body') else {}
        metadata = serwoObject.get_metadata() if hasattr(serwoObject, 'get_metadata') else {}
        
        node_metrics = body.get("node_metrics", [])
        node_metrics.append({
            "node": "ProcessAllSubproblems",
            "status": "failed",
            "error": f"Unhandled exception: {str(e)}"
        })
        body["node_metrics"] = node_metrics
        body["workflow_path"] = body.get("workflow_path", []) + ["ProcessAllSubproblems (failed)"]
        body["next_node_name"] = "AssembleFinalCode"  # Continue workflow even on failure
        
        return SerWOObject(body=body, metadata=metadata)