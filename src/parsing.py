from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict
import json
from pydantic import BaseModel, model_validator
from typing import Dict
import json


class Prompt(BaseModel):
    prompt: str

    @model_validator(mode="after")
    def check_prompt(self):
        if not self.prompt.strip():
            raise ValueError("invalid")
        return self


class Parameter(BaseModel):
    type: str

    @model_validator(mode="after")
    def check_type(self):
        if not self.type.strip():
            raise ValueError("invalid")
        return self


class Returns(BaseModel):
    type: str

    @model_validator(mode="after")
    def check_type(self):
        if not self.type.strip():
            raise ValueError("invalid")
        return self


class Functions(BaseModel):
    name: str 
    description: str 
    parameters: Dict[str, Parameter]
    returns: Returns 





def main():
    with open("/goinfre/cramadan/project/data/input/function_calling_tests.json", "r") as file:
        data = json.load(file)
    prompts = []
    i = 0
    while i < len(data):
        prompt = Prompt.model_validate(data[i])
        prompts.append(prompt)
        i += 1
    # for prompt in prompts:
    #     print(prompt.prompt)
    with open("/goinfre/cramadan/project/data/input/functions_definition.json", "r") as file:
        data_function = json.load(file)
    functions = []
    i = 0
    while i < len(data_function):
        func = Functions.model_validate(data_function[i])
        functions.append(func)
        i+=1
    # for function in functions:
    #     print(function.name)
    #     print(function.parameters["a"]["type"])
    #     print(function.parameters["b"]["type"])
    #     print(function.returns["type"])

try:
    main()
except Exception as e:
    print("Error:", e)
        # return 