import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio
from google import genai
from concurrent.futures import TimeoutError
from functools import partial
import json

# Load environment variables from .env file
load_dotenv()

# Access your API key and initialize Gemini client correctly
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

max_iterations = 6
last_response = None
iteration = 0
iteration_response = []

EVALUATE_PROMPT = """You are a Prompt Evaluation Assistant.

You will receive a prompt written by a student. Your job is to review this prompt and assess how well it supports structured, step-by-step reasoning in an LLM (e.g., for math, logic, planning, or tool use).

Evaluate the prompt on the following criteria:

1. ✅ Explicit Reasoning Instructions  
   - Does the prompt tell the model to reason step-by-step?  
   - Does it include instructions like “explain your thinking” or “think before you answer”?

2. ✅ Structured Output Format  
   - Does the prompt enforce a predictable output format (e.g., FUNCTION_CALL, JSON, numbered steps)?  
   - Is the output easy to parse or validate?

3. ✅ Separation of Reasoning and Tools  
   - Are reasoning steps clearly separated from computation or tool-use steps?  
   - Is it clear when to calculate, when to verify, when to reason?

4. ✅ Conversation Loop Support  
   - Could this prompt work in a back-and-forth (multi-turn) setting?  
   - Is there a way to update the context with results from previous steps?

5. ✅ Instructional Framing  
   - Are there examples of desired behavior or “formats” to follow?  
   - Does the prompt define exactly how responses should look?

6. ✅ Internal Self-Checks  
   - Does the prompt instruct the model to self-verify or sanity-check intermediate steps?

7. ✅ Reasoning Type Awareness  
   - Does the prompt encourage the model to tag or identify the type of reasoning used (e.g., arithmetic, logic, lookup)?

8. ✅ Error Handling or Fallbacks  
   - Does the prompt specify what to do if an answer is uncertain, a tool fails, or the model is unsure?

9. ✅ Overall Clarity and Robustness  
   - Is the prompt easy to follow?  
   - Is it likely to reduce hallucination and drift?

---
Respond with the exact structured output in this format based on above information provided:
{
  "explicit_reasoning": true,
  "structured_output": true,
  "tool_separation": true,
  "conversation_loop": true,
  "instructional_framing": true,
  "internal_self_checks": false,
  "reasoning_type_awareness": false,
  "fallbacks": false,
  "overall_clarity": here, give a short description of the clarity of the prompt
}
"""

async def generate_with_timeout(client, prompt, timeout=10):
    """Generate content with a timeout"""
    print("Starting LLM generation...")
    try:
        # Convert the synchronous generate_content call to run in a thread
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None, 
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            ),
            timeout=timeout
        )
        print("LLM generation completed")
        return response
    except TimeoutError:
        print("LLM generation timed out!")
        raise
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        raise

def reset_state():
    """Reset all global variables to their initial state"""
    global last_response, iteration, iteration_response
    last_response = None
    iteration = 0
    iteration_response = []


async def main():
    reset_state()
    print("Starting main execution...")
    try:
        server_params = StdioServerParameters(
            command="python",
            args=["actual_assignment\example2-3.py"]
        )

        async with stdio_client(server_params) as (read, write):
            print("Connection established, creating session...")
            async with ClientSession(read, write) as session:
                print("Session created, initializing...")
                await session.initialize()

                print("Requesting tool list...")
                tools_result = await session.list_tools()
                tools = tools_result.tools

                tools_description = []
                for i, tool in enumerate(tools):
                    try:
                        params = tool.inputSchema
                        desc = getattr(tool, 'description', 'No description available')
                        name = getattr(tool, 'name', f'tool_{i}')

                        if 'properties' in params:
                            param_details = []
                            for param_name, param_info in params['properties'].items():
                                param_type = param_info.get('type', 'unknown')
                                param_details.append(f"{param_name}: {param_type}")
                            params_str = ', '.join(param_details)
                        else:
                            params_str = 'no parameters'

                        tool_desc = f"{i+1}. {name}({params_str}) - {desc}"
                        tools_description.append(tool_desc)
                    except Exception as e:
                        tools_description.append(f"{i+1}. Error processing tool")

                tools_description = "\n".join(tools_description)

                system_prompt = f"""You are a math and drawing agent solving problems in steps using provided tools.

Available tools:
{tools_description}

Respond with EXACTLY ONE line in one of these formats:
1. For function calls:
   FUNCTION_CALL: function_name|param1|param2|...
2. For final answers:
   FINAL_ANSWER: [number]

Rules:
- You must perform all computations and drawings using available tools.
- Only give FINAL_ANSWER when all work is complete.
- Don't repeat a function call with the same inputs.
- When returning multiple values, process them properly.

Examples:
- FUNCTION_CALL: add|5|3
- FUNCTION_CALL: strings_to_chars_to_int|INDIA
- FUNCTION_CALL: open_paint
- FINAL_ANSWER: [42]

Additional knowledge:
- For drawing the rectangle, you start the co-ordinates from  x1 580, y1 400, x2 830, y2 600 
"""

                query = """Find the ASCII values of characters in INDIA and then return sum of exponentials of those values. Once done, DO THE FOLLOWING: open Paint, then draw a rectangle, and then write the final answer inside it."""

                global iteration, last_response
                while iteration < max_iterations:
                    print(f"\n--- Iteration {iteration + 1} ---")
                    if last_response is None:
                        current_query = query
                    else:
                        current_query = current_query + "\n\n" + " ".join(iteration_response)
                        current_query = current_query + "  What should I do next?"

                    prompt = f"{system_prompt}\n\nQuery: {current_query} "
                    eval_abv_prmt = f"{EVALUATE_PROMPT}\n\n SYSTEM PROMPT: {system_prompt}\n\nQuery: {current_query}"
                    get_response = await generate_with_timeout(client, eval_abv_prmt)
                    print(f"Response: {get_response.text.strip()}")
                    clean_res = get_response.text.strip().replace("```json", "").replace("```", "")
                   
                    python_dict = json.loads(clean_res)
                    print(python_dict)
                    if python_dict['structured_output'] == True:
                        try:
                            response = await generate_with_timeout(client, prompt)
                            response_text = response.text.strip()
                            print(f"LLM Response: {response_text}")
                            for line in response_text.split('\n'):
                                if line.strip().startswith("FUNCTION_CALL:"):
                                    response_text = line.strip()
                                    break
                        except Exception as e:
                            print(f"Failed to get LLM response: {e}")
                            print(f"!!!!!! HEY FINE TUNE THE PROMPT FOR STRUCTURED OUTPUT !!!!!")
                            break

                    if response_text.startswith("FUNCTION_CALL:"):
                        _, function_info = response_text.split(":", 1)
                        parts = [p.strip() for p in function_info.split("|")]
                        func_name, params = parts[0], parts[1:]

                        try:
                            tool = next((t for t in tools if t.name == func_name), None)
                            if not tool:
                                raise ValueError(f"Unknown tool: {func_name}")

                            arguments = {}
                            schema_properties = tool.inputSchema.get('properties', {})

                            for param_name, param_info in schema_properties.items():
                                if not params:
                                    raise ValueError(f"Not enough parameters for {func_name}")

                                value = params.pop(0)
                                param_type = param_info.get('type', 'string')

                                if param_type == 'integer':
                                    arguments[param_name] = int(value)
                                elif param_type == 'number':
                                    arguments[param_name] = float(value)
                                elif param_type == 'array':
                                    if isinstance(value, str):
                                        value = value.strip('[]').split(',')
                                    arguments[param_name] = [int(x.strip()) for x in value]
                                else:
                                    arguments[param_name] = str(value)

                            result = await session.call_tool(func_name, arguments=arguments)
                            if hasattr(result, 'content'):
                                if isinstance(result.content, list):
                                    iteration_result = [item.text if hasattr(item, 'text') else str(item) for item in result.content]
                                else:
                                    iteration_result = str(result.content)
                            else:
                                iteration_result = str(result)

                            if isinstance(iteration_result, list):
                                result_str = f"[{', '.join(iteration_result)}]"
                            else:
                                result_str = str(iteration_result)

                            iteration_response.append(
                                f"In iteration {iteration + 1}, called {func_name} with {arguments}, result: {result_str}."
                            )
                            last_response = iteration_result

                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            iteration_response.append(f"Error in iteration {iteration + 1}: {str(e)}")
                            break

                    elif response_text.startswith("FINAL_ANSWER:"):
                        print("\n=== Agent Execution Complete ===")
                        iteration_response.append(f"Final result: {response_text}")
                        break

                    iteration += 1
    except Exception as e:
        print(f"Unhandled exception: {e}")


if __name__ == "__main__":
    asyncio.run(main())
    
    
