from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_sdk.llm_sdk import Small_LLM_Model
import torch
import json


class LLM:
    def __init__(self):
        pass
    def register_function(self):
        pass
    def add_numbers(a: int, b: int):
        return a + b
    def generate_text(self,prompt,llm):
        
        inputs = llm.encode(prompt)
        inputs = inputs.tolist()[0]
        new_token =[]
        for _ in range(60):
            # print("heeeeeeeeere ",type(inputs))
            logits = llm.get_logits_from_input_ids(inputs)
            logits = torch.tensor(logits)
            predicted_tensor = torch.argmax(logits)
            new_token.append(predicted_tensor.item())
            inputs.append(predicted_tensor.item())
        answer = llm.decode(new_token)
        result = json.loads(answer)
        print(type(result))
        print(answer)






S = LLM()
llm = Small_LLM_Model()
with open('/goinfre/cramadan/project/data/input/function_calling_tests.json','r') as file:
    content = file.read()
    prompt = json.loads(content)

with open('/goinfre/cramadan/project/data/input/functions_definition.json','r') as file:
    functions_text = file.read()
    # functions = json.loads(content)
    # functions_text = json.dumps(functions, indent=2)

user_request = prompt[0]["prompt"]
S.generate_text(f"""You are a function-calling assistant.

You are given:

1. A list of available functions in JSON format.
2. A user's request.

Your task is to determine:
- which function should be called,
- and what arguments should be passed to it.

Available Functions:

{functions_text}

----------------------------------------

User Request:

{user_request}

----------------------------------------

{{
  "function": "<function_name>",
  "arguments": {{
    ...
  }}
}}

Do not explain your reasoning.
Do not return Markdown.
If no function matches, return null.""",llm)
# """
#         You are a function selector.

#         Your task:
#         Given a user request, select the SINGLE best matching function.

#         OUTPUT FORMAT:
        
#     Return ONLY the function name.
#     Plain text only.
#     No markdown.
#     No JSON.
#     No explanations.
#     No reasoning.
#     No extra spaces.
#     No punctuation.

#             SELECTION RULES:
            
#     Select ONLY from AVAILABLE FUNCTIONS.
#     Never invent function names.
#     Choose the MOST specific matching function.
#     If multiple functions could match, select the closest semantic
#     match.
#     Ignore irrelevant details in the request.
#     Match intent, not exact wording.

#             FAILURE RULES:
            
#     Empty request → null
#     No suitable function → null
#     Ambiguous request → null
#     Multiple unrelated tasks → null

#             AVAILABLE FUNCTIONS:
# """

