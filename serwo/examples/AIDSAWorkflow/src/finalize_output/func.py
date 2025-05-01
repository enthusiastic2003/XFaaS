# src/finalize_output/func.py
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

NODE_MEMORY_MB = 128

def function(serwoObject) -> SerWOObject:
    node_start_time = time.time()
    body = serwoObject.get_body()
    metadata = serwoObject.get_metadata()

    # Extract final state
    final_code = body.get("final_assembled_code")
    assembly_status = body.get("assembly_status", "unknown")
    assembly_error = body.get("assembly_error")
    workflow_path_list = body.get("workflow_path", [])
    node_metrics_list = body.get("node_metrics", [])
    total_duration = body.get("total_duration_sec", 0.0)
    total_cost = body.get("total_cost", 0.0)

    # Prepare the final output structure
    final_output = {
        "problem_description": body.get("problem_description"),
        "assembly_status": assembly_status,
        "final_code": final_code if final_code else f"Assembly failed: {assembly_error}",
        "execution_summary": {
            "workflow_path": " -> ".join(workflow_path_list),
            "total_duration_sec": total_duration,
            "total_estimated_cost": total_cost,
            "node_metrics": node_metrics_list
        }
    }
    # Add metrics for this node
    node_end_time = time.time()
    node_duration_sec = node_end_time - node_start_time
    node_cost = calculate_cost(NODE_MEMORY_MB, node_duration_sec) # Minimal cost

    final_output["execution_summary"]["total_duration_sec"] += node_duration_sec
    final_output["execution_summary"]["total_estimated_cost"] += node_cost
    final_output["execution_summary"]["node_metrics"].append({
        "node": "FinalizeOutput", "duration_sec": node_duration_sec, "cost": node_cost, "memory_mb": NODE_MEMORY_MB
    })
    final_output["execution_summary"]["workflow_path"] += " -> FinalizeOutput"

    print(f"FinalizeOutput: Formatting completed. Total Duration={final_output['execution_summary']['total_duration_sec']:.3f}s, Total Cost=${final_output['execution_summary']['total_estimated_cost']:.6f}")

    # Return the structured final output as the body
    return SerWOObject(body=final_output, metadata=metadata)
