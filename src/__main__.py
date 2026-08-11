from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_sdk.llm_sdk import Small_LLM_Model
import torch
import json


class LLM:
    def __init__(self, llm):
        with open('/goinfre/cramadan/project/data/input/functions_definition.json', 'r') as file:
            self.functions = json.load(file)

        self.fixed_tokens = {
            "{": llm.encode("{").tolist()[0],
            "}": llm.encode("}").tolist()[0],
            '"': llm.encode('"').tolist()[0],
            ":": llm.encode(":").tolist()[0],
            ",": llm.encode(",").tolist()[0],
            "function": llm.encode("function").tolist()[0],
            "arguments": llm.encode("arguments").tolist()[0]
        }

    def register_function(self):
        pass
    def add_numbers(a: int, b: int):
        return a + b
    def all_functions(self):
        lst = []

        for function in self.functions:
            lst.append(function["name"])

        return lst
    
    
   
    def parameter_type_func(self,parameter):
        lst = "l"
        # lst = []
        # for function in functions_text:
        for function in self.functions:
            try:
                lst = function["parameters"][parameter]
            except:
                continue


        lst = lst["type"]
        return lst


    def get_parameters(self, function_name):
        for function in self.functions:
            if function["name"] == function_name:
                return function["parameters"]

        return None    
    def ft_constrain_one_token(self,parameter,inputs,new_token):
        logits = llm.get_logits_from_input_ids(inputs)
        logits = torch.tensor(logits)
        wanted = llm.encode(parameter).tolist()[0][0]
        
        original_logits = logits.clone()
        logits[:] = float("-inf")
        logits[wanted] = original_logits[wanted]
        
        # for i in range(len(logits)):
        #     if i not in [wanted]:
        #         logits[i] = float("-inf")
        predicted_tensor = torch.argmax(logits)
        new_token.append(predicted_tensor.item())
        inputs.append(predicted_tensor.item())
        return new_token,inputs
    
    def ft_constrain_tokens(self, parameter, inputs, new_token):
        ids = parameter

        for token_id in ids:
            new_token.append(token_id)
            inputs.append(token_id)

        return new_token, inputs

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
            original_logits = logits.clone()
            logits[:] = float("-inf")
            logits[lst_index] = original_logits[lst_index]
            # for n in range(len(logits)):
            #     if n not in lst_index:
            #         logits[n] = float("-inf")
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
            # for n in range(len(logits)):
            #     if n not in lst_index:
            #         logits[n] = float("-inf")

            original_logits = logits.clone()
            logits[:] = float("-inf")
            logits[lst_index] = original_logits[lst_index]
            predicted_tensor = torch.argmax(logits)

            for fun in ids_lst:
                predicted = predicted_tensor.item()
                if len(fun) == 0 or 0 >= len(fun) or fun[0] != predicted:
                    remove_lst.append(fun) 
                else:
                    fun.pop(0)
# numpy
            for fun in remove_lst:
                if fun not in ids_lst:
                    continue
                ids_lst.remove(fun)
            name_of_parameter.append(predicted_tensor.item())
            new_token.append(predicted_tensor.item())
            inputs.append(predicted_tensor.item())
        return new_token,inputs,name_of_parameter
    def ft_numb(self,inputs,new_token):
        for _ in range(20):
            logits = llm.get_logits_from_input_ids(inputs)
            logits = torch.tensor(logits)

            predicted_tensor = torch.argmax(logits)
            token_text = llm.decode([predicted_tensor.item()])
            
            if "," in token_text or "}" in token_text:
                break
            
            new_token.append(predicted_tensor.item())
            inputs.append(predicted_tensor.item())

        return new_token,inputs
    def ft_string(self,inputs,new_token):
        # print("the input that will",llm.decode(inputs))
        stop = self.fixed_tokens['"']
        for _ in range(30):
            logits = llm.get_logits_from_input_ids(inputs)
            logits = torch.tensor(logits)

            predicted_tensor = torch.argmax(logits)
            token_id = predicted_tensor.item()

            token_text = llm.decode([token_id])

            # print("---> predict out", repr(token_text))
            if '"' in token_text and "\n" in token_text:
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens['"'],inputs,new_token)
                # print("the first break")
                break
            new_token.append(token_id)
            inputs.append(token_id)

            if '"' in token_text:
                # print("because it stop here")
                break



        return new_token, inputs
    def generate_text(self,prompt,llm):
        inputs = llm.encode(prompt)
        inputs = inputs.tolist()[0]
        new_token =[]
        start = 0
        for _ in range(60):
            if start == 0:
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens["{"],inputs,new_token)
                start +=1
            elif start ==1:
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens['"'],inputs,new_token)
                new_token,inputs = self.ft_constrain_tokens(self.fixed_tokens["function"],inputs,new_token)
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens['"'],inputs,new_token)
                start +=1
            elif start == 2:
                new_token,inputs = self.ft_constrain_tokens(self.fixed_tokens[":"],inputs,new_token)
                start +=1
            elif start == 3:
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens['"'],inputs,new_token)
                new_token,inputs,name_of_func = self.ft_constrain_name_function(inputs,new_token)
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens['"'],inputs,new_token)   
                start+=1
            elif start == 4:
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens[','],inputs,new_token)
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens['"'],inputs,new_token)
                new_token,inputs = self.ft_constrain_tokens(self.fixed_tokens["arguments"],inputs,new_token)
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens['"'],inputs,new_token)
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens[':'],inputs,new_token)
                start +=1
            elif start == 5:
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens["{"],inputs,new_token)
                start +=1
            elif start == 6:
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens['"'],inputs,new_token)
                new_token,inputs,parameter = self.ft_constrain_parameters(inputs,new_token,name_of_func)

                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens['"'],inputs,new_token)
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens[':'],inputs,new_token)
                start+=1
            elif start == 7:
                parameters_t = llm.decode(parameter).strip()
                parameter_type = self.parameter_type_func(parameters_t).strip()
                if parameter_type == "number":
                    new_token,inputs = self.ft_numb(inputs,new_token)
                    
                elif parameter_type == "string":
                    # print("it go here string correctly")
                    new_token, inputs = self.ft_constrain_tokens(self.fixed_tokens['"'], inputs, new_token)
                    new_token,inputs = self.ft_string(inputs,new_token)
                    # print(llm.decode(new_token))
                else:
                    print("None parameter")
                start+=1

            elif start == 8:
                logits = torch.tensor(llm.get_logits_from_input_ids(inputs))
                comma = self.fixed_tokens[","][0]
                brace = self.fixed_tokens["}"][0]

                original_logits = logits.clone()
                logits[:] = float("-inf")
                logits[brace] = original_logits[brace]
                logits[comma] = original_logits[comma]

                # for i in range(len(logits)):
                #     if i not in [comma, brace]:
                #         logits[i] = float("-inf")
                predicted_tensor = torch.argmax(logits)
                token = llm.decode([predicted_tensor.item()])
                # print("stop here first")
                # print("it reached hereee",llm.decode(new_token))
                start += 1
            elif start == 9:
                nw = llm.decode(new_token)
                if token == ",":
                    new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens[","],inputs,new_token)
                    start = 6
                elif token == "}" and nw[-1] == ",":
                    start = 6
                elif token == "}":
                    start +=1
                else:
                    print("Invalid separator:", token)
                    break
                # start += 1
            elif start == 10:
                # print("before the state =10 ",llm.decode(new_token))
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens["}"],inputs,new_token)
                new_token,inputs =self.ft_constrain_tokens(self.fixed_tokens["}"],inputs,new_token)
                # print("after the state =10 ",llm.decode(new_token))
                start +=1
            else:
                break
                
      
        
        
        answer = llm.decode(new_token)
        print(repr(answer))
   




llm = Small_LLM_Model()
S = LLM(llm)
# laaalm = Small_LLM_Model()
# S = LLM(laaalm)
with open('/goinfre/cramadan/project/data/input/function_calling_tests.json','r') as file:
    content = file.read()
    prompt = json.loads(content)

with open('/goinfre/cramadan/project/data/input/functions_definition.json','r') as file:
    functions_text = file.read()
    # functions = json.loads(content)
    # functions_text = json.dumps(functions, indent=2)

for i in range(11):
    print("--------------->the promopt",i)
    user_request = prompt[i]["prompt"]    
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
