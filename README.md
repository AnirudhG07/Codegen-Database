# Codegen-Database

This repository contains codes generated by Salesforce/Codegen model on Datasets like HumanEval Dataset.<br> The dataset can be used for research, analysis, benchmarking, etc. purposes. 

# Key features 

## Parameters Fixed
1) Temparture=0 is set
2) do_sample=True

## Parameter Variation
1) max tokens- 128, 256, 512
2) Model variation- codegen-350M, codegen-2B, codegen-6B
3) Model type- multi, mono, nl

## About
The dataset is generated by running on above parameters. LLM's like codegen these are used for generating basic codes in place of coders to increase work efficiency. This dataset promotes research for this LLM Codegen and its various versions. You may use these datasets freely for any purpose.

**Note:** 
- You may notice some codes to be wrong, incomplete, not properly indented, etc. These are as it is codes produced by codegen. To be able to test the abilities of codegen, the presence of these errors are reassuring the validity of this database.
- Since more than 1 code may be produced for 512 tokens, only the first one will be considered. If the docstring it too large, the code may not get produced, even the docstring may not be fully generated since token generated by model > max tokens.
- `do_sample=True` is set for data analysis and research as it prevents extra unwanted variation and generates the most likely code. 

# Salesforce/Codegen model
CodeGen is a family of autoregressive language models for program synthesis from the paper: A Conversational Paradigm for Program Synthesis by Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, Caiming Xiong.<br> The models are originally released in this repository, under 3 pre-training data variants (NL, Multi, Mono) and 4 model size variants (350M, 2B, 6B, 16B).
<br>
Codegen produces python codes on providing appropriate docstring as model prompts.<br>
Visit codegen on <a href="https://huggingface.co/">Hugging Face</a> <a href="https://huggingface.co/Salesforce/codegen-2B-mono"> Codegen </a>.

# HumanEval Dataset
The `HumanEval` dataset is an infamous set of docstrings made to test functioning of code models made, and rate them based on criterion set by them and you. Visit the HumanEval GitHub Repository for more information. <br>
This repository contains the HumanEval.json file containing those prompts systematically, and their codes generated respectively.
<br>

-- For ease of accessing file inside this repository, please refer the following to link to access set of well-named zip files (containing 164 codes) and download the required one.

|  Codegen |   Version    |    link for dataset    |
| :-----: | :------------------: | :--------------: |
| 2B |     mono, multi, nl     |      https://github.com/AnirudhG07/Codegen-Database/tree/main/HumanEval/Codegen-2B      | 
| 350M |     mono, multi, nl     |      https://github.com/AnirudhG07/Codegen-Database/tree/main/HumanEval/Codegen-350M     | 
| 6B |     mono, multi, nl     |      https://github.com/AnirudhG07/Codegen-Database/tree/main/HumanEval/Codegen-6B     | 

<br>
To compare with gpt-3.5-turbo, 164 codes have also been added <a href="https://github.com/AnirudhG07/Codegen-Database/blob/main/HumanEval/gpt_3.5_turbo_codes.zip"> here </a> . Note: The GPT codes may themselves may not be the correct solution. <br>

**PARAMETERS FOR THE GPT CODES**:
<ul>
- Temperature = 0<br>
- max_tokens = 512<br>
- top_p = 1<br>
</ul>

<br>
Visit the official Kaggle <a href="https://www.kaggle.com/datasets/anirudhgupta1729/codegen-humaneval-database"> Codegen HumanEval Database</a> page to download codes from there too. Don't forget to upvote the dataset!



