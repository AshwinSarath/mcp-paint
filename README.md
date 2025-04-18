# Prompt Evaluation Assistant

## 📌 Project Purpose

Evaluate student-written prompts using an LLM (e.g., ChatGPT) based on structured reasoning criteria like step-by-step thinking, output format, and robustness.

---

## ✅ What Does This Code Do?

### 1. 🧠 Defines Evaluation Criteria Prompt

The `EVALUATE_PROMPT` clearly lists all 9 evaluation points:

```python
EVALUATE_PROMPT = """You are a Prompt Evaluation Assistant.
...
Respond with the exact structured output in this format:
{
  "explicit_reasoning": true,
  ...
  "overall_clarity": here, give a short description of the clarity of the prompt
}
"""
2. 🧩 Builds the Evaluation Query
Merges the evaluation instructions with the system prompt and student query:
eval_abv_prmt = f"{EVALUATE_PROMPT}\n\nSYSTEM PROMPT: {system_prompt}\n\nQuery: {current_query}"

3. Send Evaluation Request to LLM
Uses generate_with_timeout() to query the model:


get_response = await generate_with_timeout(client, eval_abv_prmt)
4. Parse the JSON Output
Cleans the response and loads it as a Python dict:


clean_res = get_response.text.strip().replace("```json", "").replace("```", "")
python_dict = json.loads(clean_res)
5. Check if Prompt is Structured
Only continue if the prompt has a valid output format:


if python_dict['structured_output'] == True:
6. Generate Final LLM Response
If the prompt passes evaluation, send it for full response:


response = await generate_with_timeout(client, prompt)
response_text = response.text.strip()

7. Extract FUNCTION_CALL or Final Output
Looks for a clean, structured line:


for line in response_text.split('\n'):
    if line.strip().startswith("FUNCTION_CALL:"):
        response_text = line.strip()
        break
