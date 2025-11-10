import json
import re
import time
import traceback
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

from anthropic import Anthropic

from dotenv import load_dotenv

load_dotenv()

# Embedded evaluator prompt
EVALUATION_PROMPT = """You are an AI assistant with access to tools.

⚠️ CRITICAL OUTPUT FORMAT REQUIREMENT ⚠️
You MUST wrap your entire response using these THREE XML tags in this EXACT order:

<summary>
[Your FULL explanation of steps, tools used, inputs/outputs, and reasoning]
</summary>

<feedback>
[Your feedback on the tools provided]
</feedback>

<response>
[ONLY the raw answer - EXACT format as requested, NO extra symbols or text]
</response>

CRITICAL RULES FOR <response> TAG:
1. Return ONLY the raw value - NO explanations, NO sentences, NO markdown formatting
2. Do NOT add currency symbols ($, €, £) unless specifically asked
3. If question says "in dollars" - return just the NUMBER (e.g., 11614.72, NOT $11614.72)
4. If question says "return the amount with $" - then include $ (e.g., $11614.72)
5. Match the EXACT precision requested (e.g., "2 decimal places" = 11614.72, NOT 11614.7 or 11614.721)

✅ CORRECT EXAMPLES:

Example 1 - Question: "What is the final amount in dollars?"
<summary>
I used the calculator tool to compute compound interest.
The result is 11614.72 dollars.
</summary>
<feedback>
The calculator tool lacks a description field.
</feedback>
<response>
11614.72
</response>

Example 2 - Question: "What is the population standard deviation? Round to 2 decimal places."
<summary>
I calculated the standard deviation using the formula.
The result is 7.61.
</summary>
<feedback>
Tool worked well.
</feedback>
<response>
7.61
</response>

Example 3 - Scientific notation requested
<summary>
Calculated energy using E=hc/λ formula.
Result is 3.61e-19 joules.
</summary>
<feedback>
Tool performed calculation correctly.
</feedback>
<response>
3.61e-19
</response>

❌ WRONG - Adding $ when "in dollars" is mentioned:
<response>
$11614.72
</response>

❌ WRONG - Explanation in response:
<response>
The final amount is 11614.72 (rounded to 2 decimal places)
</response>

❌ WRONG - Bold/formatting:
<response>
**11614.72**
</response>

✅ CORRECT - Just the number:
<response>
11614.72
</response>

REMEMBER: 
- "in dollars" or "in meters" means return JUST THE NUMBER
- Put ALL context and explanations in <summary>
- <response> = bare value only, exact format requested"""

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_AUTH_TOKEN"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
model = "claude-sonnet-4-5-20250929"


def agent_loop(prompt: str, tools: List[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:

    """Simplified agent class for tool evaluation"""
    messages = [{"role": "user", "content": prompt}]

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=EVALUATION_PROMPT,
        messages=messages,
        tools=tools,
    )
    messages.append({"role": "assistant", "content": response.content})

    # Track tool calls with timing
    tool_metrics = {}  # {tool_name: {"count": N, "durations": [X1, X2, ...]}}

    def _prepare_tool_result(tool_use_id, tool_result):
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_result,
                }
            ],
        }

    while response.stop_reason == "tool_use":
        tool_use = next(block for block in response.content if block.type == "tool_use")
        tool_name = tool_use.name

        tool_start_ts = time.time()
        try:
            tool_response = eval(
                f"{tool_name}(**tool_use.input)"
            )  # Call the tool function with its input
        except Exception as e:
            tool_response = f"Error executing tool {tool_name}: {str(e)}\n"
            tool_response += traceback.format_exc()
        tool_duration = time.time() - tool_start_ts

        # Update tool metrics
        if tool_name not in tool_metrics:
            tool_metrics[tool_name] = {"count": 0, "durations": []}
        tool_metrics[tool_name]["count"] += 1
        tool_metrics[tool_name]["durations"].append(tool_duration)

        # Prepare tool result and append to messages
        messages.append(_prepare_tool_result(tool_use.id, tool_response))
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=EVALUATION_PROMPT,
            messages=messages,
            tools=tools,
        )
        messages.append({"role": "assistant", "content": response.content})
    
    # After tool execution is complete, add a reminder about XML format if needed
    if response.stop_reason != "tool_use":
        # Check if response contains XML tags
        response_text = next(
            (block.text for block in response.content if hasattr(block, "text")),
            "",
        )
        if not ("<summary>" in response_text and "<feedback>" in response_text and "<response>" in response_text):
            # Add a format reminder and get a new response
            messages.append({
                "role": "user",
                "content": "IMPORTANT: You must format your response using the three required XML tags: <summary>, <feedback>, and <response>. Please reformat your answer to include all three tags with your content inside them."
            })
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=EVALUATION_PROMPT,
                messages=messages,
                tools=tools,
            )
            messages.append({"role": "assistant", "content": response.content})

    response = next(
        (block.text for block in response.content if hasattr(block, "text")),
        None,
    )
    return response, tool_metrics

def parse_evaluation_file(file_path: Path) -> List[Dict[str, Any]]:
    """Parse XML evaluation file and return list of evaluation tasks."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        evaluations = []

        # Check for task elements
        tasks = root.findall(".//task")
        for task in tasks:
            prompt_elem = task.find("prompt")
            response_elem = task.find("response")

            if prompt_elem is not None and response_elem is not None:
                eval_dict = {
                    "prompt": (prompt_elem.text or "").strip(),
                    "response": (response_elem.text or "").strip(),
                }
                evaluations.append(eval_dict)

        return evaluations
    except Exception as e:
        print(f"Error parsing evaluation file {file_path}: {e}")
        return []
def evaluate_single_task(
    task: Dict[str, Any], tools: List[Dict[str, Any]], task_index: int
) -> Dict[str, Any]:
    """Evaluate a single task with the given tools."""
    start_time = time.time()

    # Run the task
    print(f"Task {task_index + 1}: Running task with prompt: {task['prompt']}")
    response, tool_metrics = agent_loop(task["prompt"], tools)

    # Extract all tagged content
    def _extract_xml_content(text, tag):
        print(f"text is {text}")
        if not text:
            return None
        pattern = rf"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    def _clean_response(text):
        """Clean verbose response to extract just the answer."""
        if not text:
            return text
        
        # Remove markdown bold/italics first
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # Try to extract a number (with optional currency symbol) from the text
        # This handles: $11,614.72, 11614.72, 3.61e-19, etc.
        number_pattern = r'[\$€£]?\s*-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?'
        
        # Pattern: "The answer is X" or "The final amount is X"
        match = re.search(r'(?:is|are)\s+(' + number_pattern + r')', text)
        if match:
            text = match.group(1)
        else:
            # Pattern: "X (rounded to...)" -> extract X from beginning
            match = re.search(r'^([^\(]+?)\s*\(', text)
            if match:
                text = match.group(1).strip()
        
        # If it's multiple lines, extract from first line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines and len(lines) > 1:
            first_line = lines[0]
            match = re.search(number_pattern, first_line)
            if match:
                text = match.group(0)
        
        # If text still has non-numeric content, try to extract just the number
        if not re.match(r'^[\$€£]?\s*-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?$', text.strip()):
            match = re.search(number_pattern, text)
            if match:
                text = match.group(0)
        
        # Remove currency symbols (we want raw numbers)
        text = re.sub(r'^[\$€£]\s*', '', text)
        
        # Remove commas from numbers (11,614.72 -> 11614.72)
        text = re.sub(r'(\d),(\d)', r'\1\2', text)
        
        # Strip any remaining whitespace
        text = text.strip()
        
        return text

    extracted_response = _extract_xml_content(response, "response")
    summary = _extract_xml_content(response, "summary")
    feedback = _extract_xml_content(response, "feedback")
    
    # Handle cases where XML tags weren't found
    if extracted_response is None:
        print(f"⚠️  Warning: Task {task_index + 1} - Response not properly formatted in XML tags")
        print(f"Raw response: {response[:200]}...")
        extracted_response = "NOT_FOUND"
    else:
        # Clean up verbose responses
        cleaned = _clean_response(extracted_response)
        if cleaned != extracted_response:
            print(f"⚠️  Warning: Task {task_index + 1} - Response was verbose, cleaned from '{extracted_response[:50]}...' to '{cleaned}'")
            extracted_response = cleaned
    duration_seconds = time.time() - start_time
    task_result = task["response"]
    print(f"extracted_response is {extracted_response}, while except response is {task_result}")
    return {
        "prompt": task["prompt"],
        "expected": task["response"],
        "actual": extracted_response,
        "score": int(extracted_response == task["response"]),
        "total_duration": duration_seconds,
        "tool_calls": tool_metrics,
        "num_tool_calls": sum(len(metrics["durations"]) for metrics in tool_metrics.values()),
        "summary": summary,
        "feedback": feedback,
    }

# Report Templates
REPORT_HEADER = """
# Evaluation Report

## Summary

- **Accuracy**: {correct}/{total} ({accuracy:.1f}%)
- **Average Task Duration**: {average_duration_s:.2f}s
- **Average Tool Calls per Task**: {average_tool_calls:.2f}
- **Total Tool Calls**: {total_tool_calls}

---
"""

TASK_TEMPLATE = """
### Task

**Prompt**: {prompt}
**Ground Truth Response**: `{expected_response}`
**Actual Response**: `{actual_response}`
**Correct**: {correct_indicator}
**Duration**: {total_duration:.2f}s
**Tool Calls**: {tool_calls}

**Summary**
{summary}

**Feedback**
{feedback}

---
"""


def run_evaluation(eval_path: str, tools: List[Dict[str, Any]]) -> str:
    """
    Run evaluation with provided tools using a simple loop.

    Args:
        eval_path: Path to XML evaluation file
        tools: List of tool definitions to use for evaluation

    """
    print("🚀 Starting Evaluation")

    eval_file = Path(eval_path)

    # Parse evaluation tasks
    tasks = parse_evaluation_file(eval_file)

    print(f"📋 Loaded {len(tasks)} evaluation tasks")

    # Simple loop to run all tasks
    results = []
    for i, task in enumerate(tasks):
        print(f"Processing task {i + 1}/{len(tasks)}")
        results.append(evaluate_single_task(task, tools, i))

    # Calculate summary statistics
    correct = sum(r["score"] for r in results)
    accuracy = (correct / len(results)) * 100
    average_duration_s = sum(r["total_duration"] for r in results) / len(results)
    average_tool_calls = sum(r["num_tool_calls"] for r in results) / len(results)
    total_tool_calls = sum(r["num_tool_calls"] for r in results)

    report = REPORT_HEADER.format(
        correct=correct,
        total=len(results),
        accuracy=accuracy,
        average_duration_s=average_duration_s,
        average_tool_calls=average_tool_calls,
        total_tool_calls=total_tool_calls,
    )

    report += "".join(
        [
            TASK_TEMPLATE.format(
                prompt=task["prompt"],
                expected_response=task["response"],
                actual_response=result["actual"],
                correct_indicator="✅" if result["score"] else "❌",
                total_duration=result["total_duration"],
                tool_calls=json.dumps(result["tool_calls"], indent=2),
                summary=result["summary"] or "N/A",
                feedback=result["feedback"] or "N/A",
            )
            for task, result in zip(tasks, results)
        ]
    )
    # Join all sections into final report
    return report

def calculator(expression: str) -> str:
    """A basic calculator that performs arithmetic operations."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# Define the tool schema for the calculator
calculator_tool = {
    "name": "calculator",
    "description": "",  # An unhelpful tool description.
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "",  # An unhelpful schema description.
            }
        },
        "required": ["expression"],
    },
}

# Set the tools list
tools = [calculator_tool]

# Run evaluation
print("✅ Using calculator tool")

report = run_evaluation(eval_path="evaluation.xml", tools=tools)

print(report)