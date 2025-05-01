import time
import json
from python.src.utils.classes.commons.serwo_objects import SerWOObject
# No cost calculator needed here unless we assign a cost to this node itself



def function(serwoObject) -> SerWOObject:
    # Default Problem if none provided via input body
    DEFAULT_PROBLEM = """
    Implement a Python function `find_max_xor(nums)` that takes a list of non-negative integers `nums`
    and returns the maximum XOR value between any two numbers in the list.
    If the list has fewer than two elements, return 0.
    Provide type hints and a docstring.
    """
    body = serwoObject.get_body() if serwoObject.get_body() else {}
    metadata = serwoObject.get_metadata()
    start_node_time = time.time() # Minimal time for this node

    problem_description = body.get("problem_description", DEFAULT_PROBLEM).strip()

    print(f"StartWorkflow: Starting with Problem:\n{problem_description[:200]}...")

    # Initialize workflow state
    initial_state = {
        "problem_description": problem_description,
        "node_metrics": [],
        "workflow_path": ["StartWorkflow"],
        "total_duration_sec": 0.0,
        "total_cost": 0.0 # Start cost at 0
    }

    # Add metrics for this node (minimal cost/duration)
    node_duration = time.time() - start_node_time
    # Minimal cost for starting node
    node_cost = 0.000001
    initial_state["total_duration_sec"] += node_duration
    initial_state["total_cost"] += node_cost
    initial_state["node_metrics"].append({
        "node": "StartWorkflow", "duration_sec": node_duration, "cost": node_cost, "memory_mb": 128 # Assume base memory
    })

    obj =  SerWOObject(body=initial_state, metadata=metadata)
    obj.set_basepath(serwoObject.get_basepath())  # Using the base path from the first object

    return obj