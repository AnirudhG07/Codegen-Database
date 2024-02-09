# HOW TO USE CODEGEN Model
from transformers import AutoModelForCausalLM, AutoTokenizer

# INITIALISING THE MODEL
# The model can be 350M-multi, anyother model of Codegen available on HuggingFace.
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono") 
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-mono")

# SET PARAMETERS
temperature=1 # choose between 0 and 1 inclusive.
max_length=256 # max tokens used by model, this includes docstring tokens. prefer to use > 512 for smooth functioning.

#PROMPT
prompt="def prime(num:int): \"\"\" Check if given number is prime or not \"\"\" "

# CODE GENERATION
input_ids = tokenizer.encode(prompt, return_tensors="pt") # tensor of input id's of promopt.

generated_ids = model.generate(input_ids, max_length=max_length, pad_token_id=50256, temperature=temperature) # generates tensor of tokens which represents our code.
# keep pad_token_id = 50256, optional, if not set the model will automatically do the same. 
generated_code=tokenizer.decode(generated_ids[0]) # decodes the tokens into our code.

print(generated_code)

