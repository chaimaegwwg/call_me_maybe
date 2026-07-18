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
    def all_functions(self):
        with open('/goinfre/cramadan/project/data/input/functions_definition.json','r') as file:
            content = file.read()
            functions_text = json.loads(content)
        lst = []
        for function in functions_text:
            lst.append(function["name"])

        return lst


    def generate_text(self,prompt,llm):
        
        inputs = llm.encode(prompt)
        inputs = inputs.tolist()[0]
        new_token =[]
        start = 0
        for _ in range(60):
            logits = llm.get_logits_from_input_ids(inputs)
            logits = torch.tensor(logits)
            if start == 0:
                wanted = llm.encode("{").tolist()[0][0]
                for i in range(len(logits)):
                    if i not in [wanted]:
                        logits[i] = float("-inf")
                predicted_tensor = torch.argmax(logits)
                new_token.append(predicted_tensor.item())
                inputs.append(predicted_tensor.item())
                start +=1
            elif start ==1:
                ids =  llm.encode("function").tolist()[0]
                for token_id in ids:
                    logits = llm.get_logits_from_input_ids(inputs)
                    logits = torch.tensor(logits)
                    for i in range(len(logits)):
                        if i not in [token_id]:
                            logits[i] = float("-inf")
                    predicted_tensor = torch.argmax(logits)
                    new_token.append(predicted_tensor.item())
                    inputs.append(predicted_tensor.item())
                start +=1
            elif start == 2:
                wanted = llm.encode(".").tolist()[0][0]
                for i in range(len(logits)):
                    if i not in [wanted]:
                        logits[i] = float("-inf")
                predicted_tensor = torch.argmax(logits)
                new_token.append(predicted_tensor.item())
                inputs.append(predicted_tensor.item())
                start +=1
            elif start == 3:
                functions = self.all_functions()
                ids_lst = []
                for function in functions:
                    ids_lst.append(llm.encode(function).tolist()[0])
                id_token = []
                id_tokens = [id_s[0] for id_s in ids_lst]
                for token_id in id_tokens:
                    logits = llm.get_logits_from_input_ids(inputs)
                    logits = torch.tensor(logits)
                    for i in range(len(logits)):
                        if i not in [token_id]:
                            logits[i] = float("-inf")
                    predicted_tensor = torch.argmax(logits)
                    new_token.append(predicted_tensor.item())
                    inputs.append(predicted_tensor.item())
                for ids in ids_lst:
                    for token_id in ids[1:]:
                        if token_id != predicted_tensor:
                            break
                        logits = llm.get_logits_from_input_ids(inputs)
                        logits = torch.tensor(logits)
                        allowed == [token_id]
                        
                        for i in range(len(logits)):
                            if i not in [allowed]:
                                logits[i] = float("-inf")
                        predicted_tensor = torch.argmax(logits)
                        new_token.append(predicted_tensor.item())
                        inputs.append(predicted_tensor.item())
                start+=1
                # for ids in functions:
                #     for token_id in ids:
                #     logits = llm.get_logits_from_input_ids(inputs)
                #     logits = torch.tensor(logits)
                #     for i in range(len(logits)):
                #         if i not in ids:
                #             allowed = token_id
                #     for i in range(len(logits)):
                #         if i not in [allowed]:
                #             logits[i] = float("-inf")
                #     predicted_tensor = torch.argmax(logits)
                #     new_token.append(predicted_tensor.item())
                #     inputs.append(predicted_tensor.item())

                # ids =  llm.encode(functions).tolist()[0]
                # for token_id in ids:
                #     logits = llm.get_logits_from_input_ids(inputs)
                #     logits = torch.tensor(logits)
                #     for i in range(len(logits)):
                #         if i not in ids:
                #             allowed = token_id
                #     for i in range(len(logits)):
                #         if i not in [allowed]:
                #             logits[i] = float("-inf")
                #     predicted_tensor = torch.argmax(logits)
                #     new_token.append(predicted_tensor.item())
                #     inputs.append(predicted_tensor.item())
                # start +=1



                # for token_id, token in vocab.items():
                #     for i in range(len(logits)):
                #         for e in range(len(wanted)):
                #             if token in logits[e]:
                #                 allowed = token_id
                #         for i in range(len(logits)):
                #             if i not in allowed:
                #                 logits[i] = float("-inf")
                # if "function" in new_token:
                #     start +=1   
            
            # predicted_tensor = torch.argmax(logits)
            # new_token.append(predicted_tensor.item())
            # inputs.append(predicted_tensor.item())
        
        # start +=1
        
        answer = llm.decode(new_token)
        result = json.loads(answer)
        # print(type(result))
        # print(answer)




# for token_id in ids.tolist()[0]:
#     print(token_id, llm.decode([token_id]))

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

