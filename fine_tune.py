from prepare_data import get_dataset
# from prepare_finetune_data import *
import os
import os.path as osp
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'    # for debugging
import torch
import logging
import sys
import numpy as np
import datasets
import evaluate
from transformers import (
    HfArgumentParser, 
    set_seed,
    AutoTokenizer, 
    RobertaForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    Trainer
)
from dataclasses import dataclass, field
import wandb


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
    defining arguments and initialize argparse
'''

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """
    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use."}
    )
    
        
@dataclass
class ModelArguments:
    """
    Arguments pertaining to our model configuration.
    """
    use_posttrained_model: bool = field(
        default=False,
        metadata={"help": "Whether to use the post trained version. If False, load pretrained roberta provided by huggingface directly."}
    )
    load_model_path: str = field(
        default="roberta-base",
        metadata={"help": "The path to the pretrained/posttrained checkpoint."}
    )
        
# definition of class TrainingArguments can be found at https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py

@dataclass
class ProjectArguments:
    """
    Arguments pertaining to wandb project record.
    """
    project_name: str = field(
        default="nlpdl-final-project-basic",
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
        metadata={"help": "The path to the merged tokenizer."}
    )
    embedding_init_type: str = field(
        default="rnd",
        metadata={"help": "If using merged tokenizer, which type of initialization for new embeddings to use."}
    )
    
        

parser = HfArgumentParser([DataArguments, ModelArguments, TrainingArguments, ProjectArguments, TokenizerArguments])
data_args, model_args, training_args, project_args, tok_args = parser.parse_args_into_dataclasses()

def preprocess_function(examples):
    text = examples["text"]
    result = tokenizer(text=text, truncation=True)
    return result

def preprocess_pair_function(examples):
    text = examples["text"]
    text_pair = examples["text_pair"]
    result = tokenizer(text=text, text_pair=text_pair, truncation=True)
    return result

def compute_micro_f1(eval_prediction: EvalPrediction):
    logits = eval_prediction.predictions
    label_ids = eval_prediction.label_ids
    predictions = np.argmax(logits, axis=-1)
    metrics = evaluate.load("f1")
    result_dict = metrics.compute(predictions=predictions, references=label_ids, average="micro")
    return result_dict

def compute_accuracy(eval_prediction: EvalPrediction):
    logits = eval_prediction.predictions
    label_ids = eval_prediction.label_ids
    predictions = np.argmax(logits, axis=-1)
    metrics = evaluate.load("accuracy")
    result_dict = metrics.compute(predictions=predictions, references=label_ids)
    return result_dict

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

'''
    set several global configurations according to the arguments
'''
supported_dataset = ["chemprot", "chemprot_v2", "bioasq"]

if data_args.dataset_name not in supported_dataset:
    logging.error("You pass a dataset name not supported. Please choose either 'chemprot' or 'bioasq' for '--dataset_name'.")
    sys.exit()
if model_args.use_posttrained_model and model_args.load_model_path == "roberta-base":
    logging.error("You choose to use a post trained model but do not provide a checkpoint file different from the default roberta. Please specify '--load_model_path'.")
    sys.exit()
if tok_args.use_merged_tokenizer and tok_args.load_tok_path == "roberta-base":
    logging.error("You choose to use a merged tokenizer but do not provide a tokenizer different from the default roberta tokenizer. Please specify '--load_tok_path'.")
    sys.exit()

if data_args.dataset_name == "chemprot" or data_args.dataset_name == "chemprot_v2":
    fn = preprocess_function
    cm = compute_micro_f1
else:
    fn = preprocess_pair_function
    cm = compute_accuracy


'''
specify the project name and run name used in wandb
'''

project_name = project_args.project_name

run_name_pre = f"data_{data_args.dataset_name}_model_{osp.basename(model_args.load_model_path)}_tok_{osp.basename(tok_args.load_tok_path)}"

if tok_args.use_merged_tokenizer:
    run_name_pre += f"_{tok_args.embedding_init_type}"

'''
    load datasets
'''

dataset = get_dataset(data_args.dataset_name)

# count the number of labels
label_list = dataset["train"].unique("labels")
num_labels = len(label_list)


'''
    run the same training script five times and report the average result
'''    

sum_metric = 0

seeds = [111111, 222222, 333333, 444444, 555555]

for multi in range(5):

    set_seed(seeds[multi])
    
    run = wandb.init(project=project_name, name=f"{run_name_pre}_times_{multi+1}", reinit=True)

    assert run is wandb.run

    wandb.config.update(data_args)
    wandb.config.update(model_args)
    wandb.config.update(training_args)
    wandb.config.update(tok_args)
   
    '''
        load the model and tokenizer
    '''
    # Create the tokenizer from a trained one
    tokenizer = AutoTokenizer.from_pretrained(tok_args.load_tok_path)

    tok_vocab_size = len(tokenizer)
    
    model = RobertaForSequenceClassification.from_pretrained(model_args.load_model_path, num_labels=num_labels)
    
    model_vocab_size = model.roberta.embeddings.word_embeddings.weight.shape[0]
   
    if tok_vocab_size > model_vocab_size:
        num_added_toks = tok_vocab_size - model_vocab_size
        model.resize_token_embeddings(tok_vocab_size)
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
                print("initialization type not supported; use random initialization by default")
    
    if data_args.dataset_name == 'chemprot':
        model, tokenizer = add_special_token(model, tokenizer, ['@GENE$', '@CHEMICAL$'])
   

    '''
        process datasets and build up datacollator
    '''
    
    dataset = dataset.map(
                fn,
                batched=True,
                desc="Running tokenizer on dataset",
            )

    # Data collator will dynamically pad the inputs received.
    data_collator = DataCollatorWithPadding(tokenizer)

    # initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=cm,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # training!
    train_result = trainer.train()
    train_metric = train_result.metrics
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", train_metric)
    trainer.save_metrics("train", train_metric)
    trainer.save_state()
    
    # evaluating!
    test_metrics = trainer.evaluate()
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    # log average result
    if data_args.dataset_name == "chemprot" or data_args.dataset_name == "chemprot_v2":
        sum_metric += test_metrics["eval_f1"]
        if multi == 4:
            wandb.log({"avg f1": sum_metric / (multi + 1)})
    else:
        sum_metric += test_metrics["eval_accuracy"]
        if multi == 4:
            wandb.log({"avg accurcy": sum_metric / (multi + 1)})
    run.finish()

wandb.finish()

