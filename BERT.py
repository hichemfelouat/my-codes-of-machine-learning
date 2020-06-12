# !pip install transformers
import torch
import transformers as ppb # pytorch transformers

# model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Want BERT instead of distilBERT? Uncomment the following line:
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, "bert-base-uncased")

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
print("------------------------------------------------------------------------")
# Convert Text to Tokens
sample_txt = "Natural language processing"
tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Sentence : ",sample_txt)
print("Tokens : ",tokens)
print("Token IDs: ", token_ids)
print("------------------------------------------------------------------------")
encoding = tokenizer.encode_plus(
    sample_txt,
    max_length=10,
    add_special_tokens=True, # Add "[CLS]" and "[SEP]""
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors="pt",  # Return PyTorch tensors
)

print(len(encoding["input_ids"][0]))
print(encoding["input_ids"][0])
print(encoding["attention_mask"])
print(tokenizer.convert_ids_to_tokens(encoding["input_ids"][0]))

print("------------------------------------------------------------------------")
last_hidden_state, pooled_output = model(
    input_ids= encoding["input_ids"],
    attention_mask=encoding["attention_mask"]
)

print(last_hidden_state.shape)
print(last_hidden_state)



