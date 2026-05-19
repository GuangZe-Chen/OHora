#Test MMLU based on sft_model with 5-shot
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer,AutoModelForCausalLM

'''
#llama2
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
'''



gpu_id = 4  #Specify the GPU number when using a single GPU
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
input_files = 'mmlu_test/test_final.json' #Specify the test_file path

sft_model_path = "outputs/llama2-7B/ohora_test/lora_model/sft_lora_model" #Specify the eval model path

#model = LlamaForCausalLM.from_pretrained(sft_model_path,torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(sft_model_path,torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(sft_model_path)

'''
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
'''

'''
pad_token = tokenizer.bos_token
print(f"Padding token: {pad_token}")

word = pad_token
token_id = tokenizer.convert_tokens_to_ids(word)
print(f"Token ID for '{word}': {token_id}")

token_id = 0  # 示例 ID
token = tokenizer.convert_ids_to_tokens(token_id)
print(f"Token ID {token_id} corresponds to token: '{token}'")

judge = 1
assert judge>1
'''
tokenizer.padding_side = "left"
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)

'''
#llama2 处理
special_tokens_dict = dict()
special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

tokenizer.add_special_tokens(special_tokens_dict)
'''

model.to(device)

choices = ['A', 'B', 'C', 'D']
answer_choice_ids = [tokenizer.encode(answer_choice, add_special_tokens=False)[0] for answer_choice in choices]
#print("True Answer Label: ",answer_choice_ids)
#print("Corresponding label: ",tokenizer.decode(answer_choice_ids))

df = pd.read_json(input_files,orient='records',lines=True)
pred = []
labels = []

for i in range(len(df)):
    prompt = df.iloc[i]
    if(prompt["label"]=='A'):
        labels.append(0)
    if(prompt["label"]=='B'):
        labels.append(1)
    if(prompt["label"]=='C'):
        labels.append(2)
    if(prompt["label"]=='D'):
        labels.append(3)
    inputs = tokenizer(prompt["instruction"], padding="longest", return_tensors="pt",add_special_tokens=False).to(device)
    decoded_sentence = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    #print(inputs["input_ids"].size()[1])
    #max_length = min(inputs["input_ids"].size()[1]+2,2048)
    batch_input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    #generate_ids = model.generate(inputs.input_ids, max_length=max_length)
    batch_logits = model(batch_input_ids, attention_mask).logits[:, -1, :]
    batch_logits = batch_logits[:, answer_choice_ids]
    batch_probs = torch.softmax(batch_logits, dim=-1)
    batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
    #print(batch_prediction_indices.item(),"=?",labels[i])
    print("Now Solving Item: ",i)
    pred.append(batch_prediction_indices.item())
    '''
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    if tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][-1] == "0":
        pred.append(0)
    if tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][-1] == "1":
        pred.append(1)
    if tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][-1] == "2":
        pred.append(2)
    if tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][-1] == "3":
        pred.append(3)
    '''

array_lines = [str(num) + '\n' for num in pred]
 
# 写入文件
with open('mmlu_eval_res.txt', 'w') as file:
    file.writelines(array_lines)

count = 0
for i in range(len(pred)):
    if(pred[i]==labels[i]):
        count = count+1

acc = count/len(pred)
print("The acc of MMLU: {:3f}".format(acc))