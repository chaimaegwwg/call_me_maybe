from transformers import AutoModelForCausalLM, AutoTokenizer

class LLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.module = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    def generate_text(self,prompts):
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.module.generate(**inputs, max_new_tokens=50)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(answer)



S = LLM()
S.generate_text(["Hi , how are you","there is a story of how made chess board..."])




