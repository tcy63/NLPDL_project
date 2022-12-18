import jsonlines
import re
from datasets import (
    Dataset, 
    DatasetDict,
    concatenate_datasets
)
import json

"""
    This function is used to process the provided text in chemprot dataset.
    
    -- The original text: 
    '<< Epidermal growth factor receptor >> inhibitors currently under investigation include the small molecules [[ gefitinib ]] (Iressa, ZD1839) and erlotinib (Tarceva, OSI-774), as well as monoclonal antibodies such as cetuximab (IMC-225, Erbitux).'
    
    -- After replacing with special string: 
    '@CHEMICAL$ inhibitors currently under investigation include the small molecules @GENE$ (Iressa, ZD1839) and erlotinib (Tarceva, OSI-774), as well as monoclonal antibodies such as cetuximab (IMC-225, Erbitux).'
"""
def special_string_replace(ori_text):
    text = re.sub("<<.+>>", "@CHEMICAL$", ori_text)
    text = re.sub("\[\[.+\]\]", "@GENE$", text)
    return text
    

chemprot_v2_ltoi = {'UPREGULATOR': 0,
                 'ACTIVATOR': 0,
                 'INDIRECT-UPREGULATOR': 0,
                 'DOWNREGULATOR': 1,
                 'INHIBITOR': 1,
                 'INDIRECT-DOWNREGULATOR': 1,
                 'AGONIST': 2,
                 'AGONIST-ACTIVATOR': 2,
                 'AGONIST-INHIBITOR': 2,
                 'ANTAGONIST': 3,
                 'PRODUCT-OF': 4,
                 'SUBSTRATE': 4,
                 'SUBSTRATE_PRODUCT-OF': 4
                }

"""
    This function is used to load the chemprot dataset.
    It returns a huggingface Dataset class instance.
"""
def jsonl_to_dataset(file_path, adapt, chemprot_v2=False):
    if adapt == False:
        dataset_labels = []
    
    dataset_texts = []
    dataset = dict()
    
    for ex in jsonlines.open(file_path):
        dataset_texts.append(special_string_replace(ex["text"]))
        if adapt == False:
            dataset_labels.append(ex["label"])
    
    if adapt == False:
        labels = list(sorted(list(set(dataset_labels))))
        ltoi = {label: ind for (ind, label) in enumerate(labels)}
        if chemprot_v2:
            ltoi = chemprot_v2_ltoi
        dataset["labels"] = [ltoi[label] for label in dataset_labels]
    
    dataset["text"] = dataset_texts
    dataset = Dataset.from_dict(dataset)
    return dataset



"""
    This function is used to load the bioasq dataset.
    BioASQ provides a list of dictionaries;
    Each dictionary contains keys: "question","text","answer";
    "question" is the first input string;
    "text" is a list of strings as the second input.
"""
def json_to_dataset(file_path, adapt):
    dataset = dict()
    if adapt == False:
        dataset_labels = []
        dataset_textpairs = []
    
    dataset_texts = []
    
    with open(file_path, 'r') as f:
        dataset_list = json.load(f)
    for item in dataset_list:
        estr = ""
        dataset_texts.append(item["question"])
        if adapt == False:
            dataset_textpairs.append(estr.join(item["text"])) # transform a list of string into one string
            dataset_labels.append(item["anwser"])
        else:
            dataset_texts.append(estr.join(item["text"]))
    
    if adapt == False:
        labels = list(sorted(list(set(dataset_labels))))
        ltoi = {label: ind for (ind, label) in enumerate(labels)}
        dataset["text_pair"] = dataset_textpairs
        dataset["labels"] = [ltoi[label] for label in dataset_labels]

    dataset["text"] = dataset_texts
    dataset  = Dataset.from_dict(dataset)
    return dataset
    


"""
    This functio is used to get a huggingface provided Dataset directly from the dataset name.
"""
def get_dataset(dataset_name, adapt=False):
    dataset = DatasetDict()
    if dataset_name == "chemprot":
        train_set = jsonl_to_dataset("chemprot/train.jsonl", adapt)
        dev_set = jsonl_to_dataset("chemprot/dev.jsonl", adapt)
        test_set = jsonl_to_dataset("chemprot/test.jsonl", adapt)
        dataset["train"] = train_set
        dataset["dev"] = dev_set
        dataset["test"] = test_set
    elif dataset_name == "chemprot_v2":
        train_set = jsonl_to_dataset("chemprot/train.jsonl", adapt, chemprot_v2=True)
        dev_set = jsonl_to_dataset("chemprot/dev.jsonl", adapt, chemprot_v2=True)
        test_set = jsonl_to_dataset("chemprot/test.jsonl", adapt, chemprot_v2=True)
        dataset["train"] = train_set
        dataset["dev"] = dev_set
        dataset["test"] = test_set
    elif dataset_name == "bioasq":
        train_set = json_to_dataset("bioasq/train.json", adapt)
        test_set = json_to_dataset("bioasq/test.json", adapt)
        dataset["train"] = train_set
        dataset["test"] = test_set
    return dataset


"""
    This function is used to expand the vocab of the pretrained model.
    We need to finish this step before tokenizing our finetuning dataset.
"""
def add_special_token(model, tokenizer, special_tags):
    special_dict = dict()
    special_dict['additional_special_tokens'] = special_tags
#     num_tokens = tokenizer.get_vocab_size(with_added_tokens = True)
    num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_tokens(special_tags)
    model.resize_token_embeddings(num_tokens+num_added_tokens)    
    return model, tokenizer



def get_adapt_dataset(dataset_name):
    chem = get_dataset("chemprot", adapt=True)
    dataset_chem = concatenate_datasets([chem["train"],chem["test"],chem["dev"]])
    bio = get_dataset("bioasq", adapt=True)
    dataset_bio = concatenate_datasets([bio["train"],bio["test"]])
    
    if dataset_name == "chemprot":
        return dataset_chem
    elif dataset_name == "bioasq":
        return dataset_bio
    elif dataset_name == "both":
        dataset = concatenate_datasets([dataset_chem, dataset_bio])
        return dataset


def get_post_dataset(input_file, test_size=0.1):
    all_lines = []
    with open(input_file, "r") as reader:
        lines = reader.readlines()
        for line in lines:
            line = line.strip() # remove the blankspace or linebreak
            if len(line) == 0:
                continue
            all_lines.append(line)

    dataset = dict()
    dataset["text"] = all_lines
    dataset = Dataset.from_dict(dataset)
#     dataset = dataset.train_test_split(test_size=test_size, seed=2022)
    return dataset
