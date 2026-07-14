from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_sdk.llm_sdk import Small_LLM_Model
import torch

class LLM:
    def __init__(self):
        pass
    def generate_text(self,prompts,llm):
        for prompt in prompts:
            inputs = llm.encode(prompt)
            inputs = inputs.tolist()[0]
            for _ in range(3):
                # print("heeeeeeeeere ",type(inputs))
                logits = llm.get_logits_from_input_ids(inputs)
                logits = torch.tensor(logits)
                predicted_tensor = torch.argmax(logits)
                inputs.append(predicted_tensor.item())
            answer = llm.decode(inputs)
            print(answer)






S = LLM()
llm = Small_LLM_Model()
S.generate_text(["Hi , how are you ?"],llm)




# def generate(self,prompts):
#     for prompt in prompts:
#         for _ in range(0,3):
#             inputs = self.tokenizer(prompt, return_tensors="pt")
#             outputs = self.module.generate(**inputs, max_new_tokens=50)
#             answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#             n = ""
#             n += prompt + answer
#             print(n)
#             prompt = n
#         print(prompt)
