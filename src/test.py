import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_sdk.llm_sdk import Small_LLM_Model


class LLM:

  def __init__(self, json_path="/goinfre/cramadan/project/data/input/functions_definition.json"):
    with open(json_path, "r") as file:
      self.functions_text = json.load(file)

  def all_functions(self):
    return [function["name"] for function in self.functions_text]

  def parameter_type_func(self, parameter):
    for function in self.functions_text:
      if parameter in function.get("parameters", {}):
        return function["parameters"][parameter]["type"]
    return "string"

  def get_parameters(self, function_name):
    for function in self.functions_text:
      if function["name"] == function_name:
        return function["parameters"]
    return None

  def ft_constrain_one_token(self, parameter, inputs, new_token):
    logits = torch.tensor(llm.get_logits_from_input_ids(inputs))
    wanted = llm.encode(parameter).tolist()[0][0]

    mask = torch.full_like(logits, float("-inf"))
    mask[wanted] = logits[wanted]

    predicted_tensor = torch.argmax(mask)
    val = predicted_tensor.item()
    new_token.append(val)
    inputs.append(val)
    return new_token, inputs

  def ft_constrain_tokens(self, parameter, inputs, new_token):
    ids = llm.encode(parameter).tolist()[0]
    for token_id in ids:
      logits = torch.tensor(llm.get_logits_from_input_ids(inputs))

      mask = torch.full_like(logits, float("-inf"))
      mask[token_id] = logits[token_id]

      predicted_tensor = torch.argmax(mask)
      val = predicted_tensor.item()
      new_token.append(val)
      inputs.append(val)
    return new_token, inputs

  def ft_constrain_name_function(self, inputs, new_token):
    name_of_func = []
    functions = self.all_functions()
    lst_gath_func = [llm.encode(fn).tolist()[0] for fn in functions]

    while True:
      lst_gath_func = [f for f in lst_gath_func if len(f) > 0]
      if not lst_gath_func:
        break

      lst_index = [f[0] for f in lst_gath_func]

      logits = torch.tensor(llm.get_logits_from_input_ids(inputs))

      mask = torch.full_like(logits, float("-inf"))
      mask[lst_index] = logits[lst_index]

      predicted_tensor = torch.argmax(mask)
      predicted = predicted_tensor.item()

      next_lst = []
      for fun in lst_gath_func:
        if fun[0] == predicted:
          next_lst.append(fun[1:])
      lst_gath_func = next_lst

      name_of_func.append(predicted)
      new_token.append(predicted)
      inputs.append(predicted)

    return new_token, inputs, name_of_func

  def ft_constrain_parameters(self, inputs, new_token, name_of_func):
    name = llm.decode(name_of_func).strip()
    parameters = self.get_parameters(name)
    if parameters is None:
      return new_token, inputs, []

    name_of_parameter = []
    ids_lst = [llm.encode(p).tolist()[0] for p in parameters]

    while True:
      ids_lst = [f for f in ids_lst if len(f) > 0]
      if not ids_lst:
        break

      lst_index = [f[0] for f in ids_lst]

      logits = torch.tensor(llm.get_logits_from_input_ids(inputs))
      mask = torch.full_like(logits, float("-inf"))
      mask[lst_index] = logits[lst_index]

      predicted_tensor = torch.argmax(mask)
      predicted = predicted_tensor.item()

      next_lst = []
      for fun in ids_lst:
        if fun[0] == predicted:
          next_lst.append(fun[1:])
      ids_lst = next_lst

      name_of_parameter.append(predicted)
      new_token.append(predicted)
      inputs.append(predicted)

    return new_token, inputs, name_of_parameter

  def ft_numb(self, inputs, new_token):
    for _ in range(20):
      logits = torch.tensor(llm.get_logits_from_input_ids(inputs))
      predicted_tensor = torch.argmax(logits)
      token_text = llm.decode([predicted_tensor.item()])
      if "," in token_text or "}" in token_text:
        break
      val = predicted_tensor.item()
      new_token.append(val)
      inputs.append(val)
    return new_token, inputs

  def ft_string(self, inputs, new_token):
    for _ in range(30):
      logits = torch.tensor(llm.get_logits_from_input_ids(inputs))
      predicted_tensor = torch.argmax(logits)
      token_id = predicted_tensor.item()
      token_text = llm.decode([token_id])

      if '"' in token_text and "\n" in token_text:
        new_token, inputs = self.ft_constrain_tokens('"', inputs, new_token)
        break

      new_token.append(token_id)
      inputs.append(token_id)

      if '"' in token_text:
        break

    return new_token, inputs

  def generate_text(self, prompt, llm):
    inputs = llm.encode(prompt).tolist()[0]
    new_token = []
    start = 0

    while True:
      if start == 0:
        new_token, inputs = self.ft_constrain_tokens("{", inputs, new_token)
        start += 1
      elif start == 1:
        new_token, inputs = self.ft_constrain_tokens('"', inputs, new_token)
        new_token, inputs = self.ft_constrain_tokens(
            "function", inputs, new_token
        )
        new_token, inputs = self.ft_constrain_tokens('"', inputs, new_token)
        start += 1
      elif start == 2:
        new_token, inputs = self.ft_constrain_tokens(":", inputs, new_token)
        start += 1
      elif start == 3:
        new_token, inputs = self.ft_constrain_tokens('"', inputs, new_token)
        new_token, inputs, name_of_func = self.ft_constrain_name_function(
            inputs, new_token
        )
        new_token, inputs = self.ft_constrain_tokens('"', inputs, new_token)
        start += 1
      elif start == 4:
        new_token, inputs = self.ft_constrain_tokens(",", inputs, new_token)
        new_token, inputs = self.ft_constrain_tokens('"', inputs, new_token)
        new_token, inputs = self.ft_constrain_tokens(
            "arguments", inputs, new_token
        )
        new_token, inputs = self.ft_constrain_tokens('"', inputs, new_token)
        new_token, inputs = self.ft_constrain_tokens(":", inputs, new_token)
        start += 1
      elif start == 5:
        new_token, inputs = self.ft_constrain_tokens("{", inputs, new_token)
        start += 1
      elif start == 6:
        new_token, inputs = self.ft_constrain_tokens('"', inputs, new_token)
        new_token, inputs, parameter = self.ft_constrain_parameters(
            inputs, new_token, name_of_func
        )
        new_token, inputs = self.ft_constrain_tokens('"', inputs, new_token)
        new_token, inputs = self.ft_constrain_tokens(":", inputs, new_token)
        start += 1
      elif start == 7:
        parameters_t = llm.decode(parameter).strip()
        parameter_type = self.parameter_type_func(parameters_t).strip()
        if parameter_type == "number":
          new_token, inputs = self.ft_numb(inputs, new_token)
        elif parameter_type == "string":
          new_token, inputs = self.ft_constrain_tokens('"', inputs, new_token)
          new_token, inputs = self.ft_string(inputs, new_token)
        else:
          print("None parameter")
        start += 1
      elif start == 8:
        logits = torch.tensor(llm.get_logits_from_input_ids(inputs))
        comma = llm.encode(",").tolist()[0][0]
        brace = llm.encode("}").tolist()[0][0]

        mask = torch.full_like(logits, float("-inf"))
        mask[comma] = logits[comma]
        mask[brace] = logits[brace]

        predicted_tensor = torch.argmax(mask)
        token = llm.decode([predicted_tensor.item()])
        start += 1
      elif start == 9:
        nw = llm.decode(new_token)
        if token == ",":
          new_token, inputs = self.ft_constrain_tokens(",", inputs, new_token)
          start = 6
        elif token == "}" and nw[-1] == ",":
          start = 6
        elif token == "}":
          start += 1
        else:
          print("Invalid separator:", token)
          break
      elif start == 10:
        new_token, inputs = self.ft_constrain_tokens("}", inputs, new_token)
        new_token, inputs = self.ft_constrain_tokens("}", inputs, new_token)
        start += 1
      else:
        break

    answer = llm.decode(new_token)
    print(repr(answer))
