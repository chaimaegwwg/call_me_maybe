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
    def get_parameters(self,function_name):
        with open('/goinfre/cramadan/project/data/input/functions_definition.json','r') as file:
            content = file.read()
            functions_text = json.loads(content)
        # for function in functions_text:
        #     if function["name"] == function_name:
        #         return function["parameters"]
        print("Searching for:", repr(function_name))
        for function in functions_text:
            print("JSON contains:", repr(function["name"]))
            if function["name"] == function_name:
                print("MATCH!")
                return function["parameters"]

        return None


    def generate_text(self,prompt,llm):
        
        inputs = llm.encode(prompt)
        inputs = inputs.tolist()[0]
        new_token =[]
        start = 0
        for _ in range(60):
            if start == 0:
                logits = llm.get_logits_from_input_ids(inputs)
                logits = torch.tensor(logits)
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
                name_of_func = []
                functions = self.all_functions()
                ids_lst = []
                for function in functions:
                    ids_lst.append(llm.encode(function).tolist()[0])
                id_token = []
                id_tokens = [id_s[0] for id_s in ids_lst]

                # for token_id in id_tokens:
                logits = llm.get_logits_from_input_ids(inputs)
                logits = torch.tensor(logits)
                for i in range(len(logits)):
                    if i not in id_tokens:
                        logits[i] = float("-inf")
                predicted_tensor = torch.argmax(logits)
                name_of_func.append(predicted_tensor.item())
                new_token.append(predicted_tensor.item())
                inputs.append(predicted_tensor.item())
                # for ids in ids_lst:
                #     if ids[0] not in [predicted_tensor]:
                #         continue
                lst_gath_func =[]
                predicted = predicted_tensor.item()
                for ids in ids_lst:
                    if ids[0] != predicted:
                        continue
                    lst_gath_func.append(ids[1:])
                if not lst_gath_func:
                    print("No valid parameter")
                    break
                # [[][]]
                max_n = 0
                for i in lst_gath_func:
                    if max_n <= len(i):
                        max_n = len(i)
                lst_index = []
                for i in range(max_n):
                    remove_lst = []
                    lst_index =[]
                    for func in lst_gath_func:
                        if len(func) == 0  or i >= len(func):
                            remove_lst.append(func)
                            continue
                        lst_index.append(func[i])

                    logits = llm.get_logits_from_input_ids(inputs)
                    logits = torch.tensor(logits)
                    for n in range(len(logits)):
                        if n not in lst_index:
                            logits[n] = float("-inf")
                    predicted_tensor = torch.argmax(logits)
                    for fun in lst_gath_func:
                        predicted = predicted_tensor.item()
                        if len(fun) == 0  or i >= len(fun) or fun[i] != predicted:
                            remove_lst.append(fun) 
                    for fun in remove_lst:
                        if fun not in lst_gath_func:
                            continue
                        lst_gath_func.remove(fun)
                    name_of_func.append(predicted_tensor.item())
                    new_token.append(predicted_tensor.item())
                    inputs.append(predicted_tensor.item())
                start+=1
            elif start == 4:
                ids =  llm.encode("parameters").tolist()[0]
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
            elif start == 5:
                logits = llm.get_logits_from_input_ids(inputs)
                logits = torch.tensor(logits)
                wanted = llm.encode("{").tolist()[0][0]
                for i in range(len(logits)):
                    if i not in [wanted]:
                        logits[i] = float("-inf")
                predicted_tensor = torch.argmax(logits)
                new_token.append(predicted_tensor.item())
                inputs.append(predicted_tensor.item())
                start +=1
            elif start == 6:
                name = llm.decode(name_of_func)
                parameters = self.get_parameters(name)
                if parameters is None:
                    print("Decoded name:", repr(name))
                    return
                
                # functions = self.all_parameter()
                ids_lst = []
                for parameter in parameters:
                    ids_lst.append(llm.encode(parameter).tolist()[0])
                id_token = []
                id_tokens = [id_s[0] for id_s in ids_lst]

                # for token_id in id_tokens:
                logits = llm.get_logits_from_input_ids(inputs)
                logits = torch.tensor(logits)
                for i in range(len(logits)):
                    if i not in id_tokens:
                        logits[i] = float("-inf")
                predicted_tensor = torch.argmax(logits)
                new_token.append(predicted_tensor.item())
                inputs.append(predicted_tensor.item())
                # for ids in ids_lst:
                #     if ids[0] not in [predicted_tensor]:
                #         continue
                lst_gath_func =[]
                predicted = predicted_tensor.item()
                for ids in ids_lst:
                    if ids[0] != predicted:
                        continue
                    lst_gath_func.append(ids[1:])
                if not lst_gath_func:
                    print("No valid parameter")
                    break
                # [[][]]
                max_n = 0
                for i in lst_gath_func:
                    if max_n <= len(i):
                        max_n = len(i)
                lst_index = []
                for i in range(max_n):
                    remove_lst = []
                    lst_index =[]
                    for func in lst_gath_func:
                        if len(func) == 0  or i >= len(func):
                            remove_lst.append(func)
                            continue
                        lst_index.append(func[i])

                    logits = llm.get_logits_from_input_ids(inputs)
                    logits = torch.tensor(logits)
                    for n in range(len(logits)):
                        if n not in lst_index:
                            logits[n] = float("-inf")
                    predicted_tensor = torch.argmax(logits)
                    for fun in lst_gath_func:
                        predicted = predicted_tensor.item()
                        if len(fun) == 0  or i >= len(fun) or fun[i] != predicted:
                            remove_lst.append(fun) 
                    for fun in remove_lst:
                        if fun not in lst_gath_func:
                            continue
                        lst_gath_func.remove(fun)
                    new_token.append(predicted_tensor.item())
                    inputs.append(predicted_tensor.item())
                start+=1
            elif start == 7:
                logits = llm.get_logits_from_input_ids(inputs)
                logits = torch.tensor(logits)
                # wanted = llm.encode("{").tolist()[0][0]
                # for i in range(len(logits)):
                #     if i not in [wanted]:
                #         logits[i] = float("-inf")
                predicted_tensor = torch.argmax(logits)
                new_token.append(predicted_tensor.item())
                inputs.append(predicted_tensor.item())
                token = llm.decode([predicted_tensor.item()])

                if token in [",", "}"]:
                    start += 1
            elif start == 8:
                if token == ",":
                    start = 6
                elif token == "}":
                    start = 9
                else:
                    print("Invalid separator:", token)
                    break
            elif start == 9:
                logits = llm.get_logits_from_input_ids(inputs)
                logits = torch.tensor(logits)
                wanted = llm.encode("}").tolist()[0][0]
                for i in range(len(logits)):
                    if i not in [wanted]:
                        logits[i] = float("-inf")
                predicted_tensor = torch.argmax(logits)
                new_token.append(predicted_tensor.item())
                inputs.append(predicted_tensor.item())
                start +=1
                
      
        
        
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
