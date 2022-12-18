from transformers import (
    RobertaForMaskedLM, 
    DataCollatorForLanguageModeling, 
    HfArgumentParser, 
    TrainingArguments, 
    Trainer, 
    AutoTokenizer
)

from datasets import Dataset    
import numpy as np
import evaluate
import torch
from prepare_data import get_post_dataset, get_adapt_dataset
# from prepare_posttrain_data import get_dataset
import os.path as osp
from dataclasses import dataclass, field
from typing import Optional
import logging
import sys
import wandb


'''
    defining arguments
'''

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """
    data_file: str = field(
        default="pubmed_base.txt",
        metadata={"help": "The text file of the data to use for post training."}
    )
    dataset_name: str = field(
        default="bioasq",
        metadata={"help": "The name of the dataset to use for adapt training."}
    )
    post_type: str = field(
        default="post",
        metadata={"help": "The type of post training; post or adapt."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to our model configuration.
    """
    load_model_path: Optional[str] = field(
        default="roberta-base",
        metadata={"help": "The path to the pretrained model."}
    )
        
# definition of class TrainingArguments can be found at https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py

@dataclass
class ProjectArguments:
    """
    Arguments pertaining to wandb project record.
    """
    project_name: Optional[str] = field(
        default="nlpdl-final-project-post",
        metadata={"help": "Project name in wandb"}
    )


@dataclass
class TokenizerArguments:
    use_merged_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use the merged tokenizer."}
    )
    load_tok_path: str = field(
        default="roberta-base",
        metadata={"help": "The path to the pretrained tokenizer."}
    )
    embedding_init_type: str = field(
        default="rnd",
        metadata={"help": "If using merged tokenizer, which type of initialization to use."}
    )
  


# tokenizer will transform a batch of inputs into {"input_ids":..., "attention_mask":...}
def tokenize_function(examples):
    text = examples["text"]
    result = tokenizer(text=text, truncation=True)
    return result 


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # find indices in each row that correspond to masked words
    indices = [[i for i, x in enumerate(labels[row]) if x != -100] for row in range(len(labels))]
    # use indices to find labels in each row that correspond to masked words
    labels = [labels[row][indices[row]] for row in range(len(labels))]
    # transform the list of lists into a lists
    labels = [item for sublist in labels for item in sublist]

    predictions = [predictions[row][indices[row]] for row in range(len(predictions))]
    predictions = [item for sublist in predictions for item in sublist]
    
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    results = {}
    results['accuracy'] = acc_metric.compute(predictions=predictions, references=labels)
    results['micro_f1'] = f1_metric.compute(predictions=predictions, references=labels, average="micro")
    results['macro_f1'] = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    return results


def group_texts(examples):
    # for each key's value, transform the nested list into one list
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # "text"
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # drop the small remainder and make total length dividable by block_size
    total_length = (total_length // block_size) * block_size
    # reorganize the nested list, each list of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


'''
    initialize logging
'''
# construct the logger object on a per-module basis
logger = logging.getLogger(__name__)

# does basic configuration for the logging system
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

'''
    initialize argparse
'''
parser = HfArgumentParser([DataArguments, ModelArguments, TrainingArguments, ProjectArguments, TokenizerArguments])
data_args, model_args, training_args, project_args, tok_args = parser.parse_args_into_dataclasses()

project_name = project_args.project_name
run_name = f"data_{data_args.data_file.split('.')[0]}_model_{osp.basename(model_args.load_model_path)}_tok_{osp.basename(tok_args.load_tok_path)}"

if tok_args.use_merged_tokenizer:
    run_name += f"_{tok_args.embedding_init_type}"

run = wandb.init(project=project_name, name=run_name)

assert run is wandb.run

wandb.config.update(data_args)
wandb.config.update(model_args)
wandb.config.update(training_args)
wandb.config.update(tok_args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Device:{}".format(device))


model = RobertaForMaskedLM.from_pretrained(model_args.load_model_path)
ori_vocab_size = model.roberta.embeddings.word_embeddings.weight.shape[0]

tokenizer = AutoTokenizer.from_pretrained(tok_args.load_tok_path)
vocab_size = len(tokenizer)

if tok_args.use_merged_tokenizer and vocab_size > ori_vocab_size:
    logging.info("------New vocabulary introduced compared with the original word embedding------")
    logging.info("[MODEL] ", ori_vocab_size)
    logging.info("[TOK] ", vocab_size)
    
    num_added_toks = vocab_size - ori_vocab_size
    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.roberta.embeddings.word_embeddings.weight
    with torch.no_grad():
        if tok_args.embedding_init_type == 'rnd':
            pass
        elif tok_args.embedding_init_type == 'zero':
            model.roberta.embeddings.word_embeddings.weight[-num_added_toks:,:] = 0
        elif tok_args.embedding_init_type == 'unk':
            unk_embedding = embeddings[tokenizer.unk_token_id,:]
            new_embeddings = torch.stack(tuple((unk_embedding for _ in range(num_added_toks))), dim=0)
            model.roberta.embeddings.word_embeddings.weight[-num_added_toks:, :] = new_embeddings
        elif tok_args.embedding_init_type == 'mean':
            pre_expansion_embeddings = embeddings[:-num_added_toks,:]
            mu = torch.mean(pre_expansion_embeddings, dim=0)
            n = pre_expansion_embeddings.size()[0]
            sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
            dist = torch.distributions.multivariate_normal.MultivariateNormal(
                mu, covariance_matrix=1e-5*sigma)
            new_embeddings = torch.stack(tuple((dist.sample() for _ in range(num_added_toks))), dim=0)               
            model.roberta.embeddings.word_embeddings.weight[-num_added_toks:, :] = new_embeddings
        else:
            logging.warning("initialization type not supported;use random initialization by default")

if data_args.post_type == "post":
    dataset = get_post_dataset(data_args.data_file)
elif data_args.post_type == "adapt":
    logging.info(f"Adapt the model on {data_args.dataset_name}")
    dataset = get_adapt_dataset(data_args.dataset_name)
    

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

block_size = 128

block_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    num_proc=4,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=block_dataset,
    compute_metrics = compute_metrics,
    data_collator=data_collator,
)

train_result = trainer.train()
train_metric = train_result.metrics
trainer.save_model()
trainer.log_metrics("train", train_metric)
trainer.save_metrics("train", train_metric)
trainer.save_state()

wandb.finish()


