# import torch

# tensor = torch.tensor([1, 2, 3, 4])

# tensor = tensor * 2

# print(tensor)



def person(name, age,pom):
    print(name, age,pom)

info = {
    "name": "Ali",
    "age": 20,
    "pom":9
}

person(**info)