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
    def parameter_type_func(self,parameter):
        
        with open('/goinfre/cramadan/project/data/input/functions_definition.json','r') as file:
            content = file.read()
            functions_text = json.loads(content)

        lst = "l"
        # lst = []
        # for function in functions_text:
        for function in functions_text:
            try:
                lst = function["parameters"][parameter]
            except:
                continue


        lst = lst["type"]
        return lst


        
    def get_parameters(self,function_name):
        with open('/goinfre/cramadan/project/data/input/functions_definition.json','r') as file:
            content = file.read()
            functions_text = json.loads(content)
        for function in functions_text:
            if function["name"] == function_name:
                return function["parameters"]

        return None

    def ft_constrain_one_token(self,parameter,inputs,new_token):
        logits = llm.get_logits_from_input_ids(inputs)
        logits = torch.tensor(logits)
        wanted = llm.encode(parameter).tolist()[0][0]
        for i in range(len(logits)):
            if i not in [wanted]:
                logits[i] = float("-inf")
        predicted_tensor = torch.argmax(logits)
        new_token.append(predicted_tensor.item())
        inputs.append(predicted_tensor.item())
        return new_token,inputs
    def ft_constrain_tokens(self,parameter,inputs,new_token):
        ids =  llm.encode(parameter).tolist()[0]
        for token_id in ids:
            logits = llm.get_logits_from_input_ids(inputs)
            logits = torch.tensor(logits)
            for i in range(len(logits)):
                if i not in [token_id]:
                    logits[i] = float("-inf")
            predicted_tensor = torch.argmax(logits)
            new_token.append(predicted_tensor.item())
            inputs.append(predicted_tensor.item())
        return new_token,inputs

    def ft_constrain_name_function(self,inputs,new_token):
        name_of_func = []
        functions = self.all_functions()
        lst_gath_func = []
        for function in functions:
            lst_gath_func.append(llm.encode(function).tolist()[0])
        while True:
            remove_lst = []
            lst_index =[]
            if all(len(x) == 0 for x in lst_gath_func):
                break
            # if len(lst_gath_func) == 1 and len(lst_gath_func[0]) == 0:
            #     break
            for func in lst_gath_func:
                if len(func) <= 0:
                    remove_lst.append(func)
                    continue
                lst_index.append(func[0])

            logits = llm.get_logits_from_input_ids(inputs)
            logits = torch.tensor(logits)
            for n in range(len(logits)):
                if n not in lst_index:
                    logits[n] = float("-inf")
            predicted_tensor = torch.argmax(logits)
            for fun in lst_gath_func:
                predicted = predicted_tensor.item()
                if len(fun) == 0 or 0 >= len(fun) or fun[0] != predicted:
                    remove_lst.append(fun) 
                else:
                    fun.pop(0)

            for fun in remove_lst:
                if fun not in lst_gath_func:
                    continue
                lst_gath_func.remove(fun)
            name_of_func.append(predicted_tensor.item())
            new_token.append(predicted_tensor.item())
            inputs.append(predicted_tensor.item())
        return new_token,inputs,name_of_func
    def ft_constrain_parameters(self,inputs,new_token,name_of_func):
        name = llm.decode(name_of_func).strip()
        parameters = self.get_parameters(name)
        if parameters is None:
            return
        name_of_parameter = []
        # functions = self.all_parameter()
        ids_lst = []
        for parameter in parameters:
            ids_lst.append(llm.encode(parameter).tolist()[0])

        while True:
            remove_lst = []
            lst_index =[]
            if all(len(x) == 0 for x in ids_lst):
                break
            # if len(ids_lst) == 1 and len(ids_lst[0]) == 0:
            #     break
            for func in ids_lst:
                if len(func) <= 0:
                    remove_lst.append(func)
                    continue
                lst_index.append(func[0])

            logits = llm.get_logits_from_input_ids(inputs)
            logits = torch.tensor(logits)
            for n in range(len(logits)):
                if n not in lst_index:
                    logits[n] = float("-inf")
            predicted_tensor = torch.argmax(logits)
            for fun in ids_lst:
                predicted = predicted_tensor.item()
                if len(fun) == 0 or 0 >= len(fun) or fun[0] != predicted:
                    remove_lst.append(fun) 
                else:
                    fun.pop(0)

            for fun in remove_lst:
                if fun not in ids_lst:
                    continue
                ids_lst.remove(fun)
            name_of_parameter.append(predicted_tensor.item())
            new_token.append(predicted_tensor.item())
            inputs.append(predicted_tensor.item())
        return new_token,inputs,name_of_parameter
    def ft_numb(self,inputs,new_token):
        new = []
        for _ in range(20):
            logits = llm.get_logits_from_input_ids(inputs)
            logits = torch.tensor(logits)
            predicted_tensor = torch.argmax(logits)
            token_text = llm.decode([predicted_tensor.item()])
            if "," in token_text or "}" in token_text:
                break
            new_token.append(predicted_tensor.item())
            new.append(llm.decode(predicted_tensor))
            inputs.append(predicted_tensor.item())

        return new_token,inputs
    def ft_string(self,inputs,new_token):
        # print("the input that will",llm.decode(inputs))
        stop = llm.encode('"').tolist()[0][0]     
        generated_content = False
        for _ in range(30):
            logits = llm.get_logits_from_input_ids(inputs)
            logits = torch.tensor(logits)

            predicted_tensor = torch.argmax(logits)
            token_id = predicted_tensor.item()

            if token_id == stop and not generated_content:
                print("predict",llm.decode(token_id))
                continue
            new_token.append(token_id)
            inputs.append(token_id)
            print("---> predict out",llm.decode(token_id))
            generated_content = True
            if token_id == stop:
                print("because it stop here")
                break

        return new_token, inputs
    def generate_text(self,prompt,llm):
        inputs = llm.encode(prompt)
        inputs = inputs.tolist()[0]
        new_token =[]
        start = 0
        for _ in range(60):
            if start == 0:
                new_token,inputs =self.ft_constrain_tokens("{",inputs,new_token)
                start +=1
            elif start ==1:
                new_token,inputs =self.ft_constrain_tokens('"',inputs,new_token)
                new_token,inputs = self.ft_constrain_tokens("function",inputs,new_token)
                new_token,inputs =self.ft_constrain_tokens('"',inputs,new_token)
                start +=1
            elif start == 2:
                new_token,inputs = self.ft_constrain_tokens(":",inputs,new_token)
                start +=1
            elif start == 3:
                new_token,inputs =self.ft_constrain_tokens('"',inputs,new_token)
                new_token,inputs,name_of_func = self.ft_constrain_name_function(inputs,new_token)
                new_token,inputs =self.ft_constrain_tokens('"',inputs,new_token)   
                start+=1
            elif start == 4:
                new_token,inputs =self.ft_constrain_tokens(',',inputs,new_token)
                new_token,inputs =self.ft_constrain_tokens('"',inputs,new_token)
                new_token,inputs = self.ft_constrain_tokens("arguments",inputs,new_token)
                new_token,inputs =self.ft_constrain_tokens('"',inputs,new_token)
                new_token,inputs =self.ft_constrain_tokens(':',inputs,new_token)
                start +=1
            elif start == 5:
                new_token,inputs =self.ft_constrain_tokens("{",inputs,new_token)
                start +=1
            elif start == 6:
                new_token,inputs =self.ft_constrain_tokens('"',inputs,new_token)
                new_token,inputs,parameter = self.ft_constrain_parameters(inputs,new_token,name_of_func)

                new_token,inputs =self.ft_constrain_tokens('"',inputs,new_token)
                new_token,inputs =self.ft_constrain_tokens(':',inputs,new_token)
                start+=1
            elif start == 7:
                parameters_t = llm.decode(parameter).strip()
                parameter_type = self.parameter_type_func(parameters_t).strip()
                if parameter_type == "number":
                    new_token,inputs = self.ft_numb(inputs,new_token)
                    
                elif parameter_type == "string":
                    print("it go here string correctly")
                    new_token, inputs = self.ft_constrain_tokens('"', inputs, new_token)
                    new_token,inputs = self.ft_string(inputs,new_token)
                else:
                    print("None parameter")
                start+=1

            elif start == 8:
                logits = torch.tensor(llm.get_logits_from_input_ids(inputs))
                comma = llm.encode(",").tolist()[0][0]
                brace = llm.encode("}").tolist()[0][0]
                for i in range(len(logits)):
                    if i not in [comma, brace]:
                        logits[i] = float("-inf")
                predicted_tensor = torch.argmax(logits)
                token = llm.decode([predicted_tensor.item()])
                start += 1
            elif start == 9:
                if token == ",":
                    new_token,inputs =self.ft_constrain_tokens(',',inputs,new_token)
                    start = 6
                elif token == "}":
                    start +=1
                else:
                    print("Invalid separator:", token)
                    break
                # start += 1
            elif start == 10:
                new_token,inputs =self.ft_constrain_tokens("}",inputs,new_token)
                new_token,inputs =self.ft_constrain_tokens('}',inputs,new_token)
                start +=1
            else:
                break
                
      
        
        
        answer = llm.decode(new_token)
        print(repr(answer))
   




S = LLM()
llm = Small_LLM_Model()
with open('/goinfre/cramadan/project/data/input/function_calling_tests.json','r') as file:
    content = file.read()
    prompt = json.loads(content)

with open('/goinfre/cramadan/project/data/input/functions_definition.json','r') as file:
    functions_text = file.read()
    # functions = json.loads(content)
    # functions_text = json.dumps(functions, indent=2)

user_request = prompt[10]["prompt"]
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
