import json
import argparse
class File:
    # def __init__(self):
    #     self.dic = {}
    def func_add(self,a,b):
        return a+b
    def func_subtract(self,a,b):
        return a-b
    def func_multiply(self,a,b):
        return a*b
    def dict_hard(self):
        dic = {
            "fn_add_numbers": self.func_add,
            "fn_subtract_numbers": self.func_subtract,
            "fn_multiply_numbers": self.func_multiply
        }
        return dic

    def main(self):
        with open('exmple.json','r') as file:
            content = file.read()
            print("content",type(content),"cont")
            data = json.loads(content)
            print("here",type(data) , "data ")
        

        print(data[0]["prompt"])
        data[0]["prompt"] = "modiefie succes"
        print(data[0]["prompt"])
        with open('exmple.json','w') as jsonFile:
            json.dump(data,jsonFile, indent=4)
        dic = self.dict_hard()
        n = dic["fn_multiply_numbers"](6,5)
        print(n)


    def parsing_prompt(self):
        with open('exmple.json','r') as file:
            content = file.read()
            data = json.loads(content)
        return data
    def parsing_func(self):
        with open('function_definition.json','r') as file:
            content = file.read()
            data = json.loads(content)
            # print(data)
        return data
    def check_func_name_dic(self):
        data = self.parsing_func()
        dic = self.dict_hard()
        print("len",len(data))
        for i in range(len(data)):
            if not data[i]["name"] in dic:
                print("not found",data[i]["name"])
            else:
                print("I found",data[i]["name"])
                

#mission now is install the package sdk 




            
#the next mission is the 
s = File()
# s.parsing_prompt()
# s.parsing_func()
s.check_func_name_dic()
# func_add()