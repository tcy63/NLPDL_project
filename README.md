# README

# Medical-Domain-Specific Large Language Model 

This repository is the final project for course NLPDL, 2022 Fall.

## Requirements

```bash
pip install -r requirements.txt
```

## Overview

### Motivation

Large-scale pre-trained language models have achieved remarkable success in the past few years. However, most pre -training methods use general corpus, such as Wikipedia. It has been proved that language models pre-trained on general corpus perform badly on domain-specific downstream tasks. So, this project aims to make pre-trained language models achieve better performance on domain-specific downstream tasks.

### Method

In this project, we focus on the medical domain. 

The RoBERTa [[1]](#1) model is loaded from huggingface as required.

The performance is measured on two downstream tasks, ChemProt and BioASQ.

1. post-training

   We use domain-specific data from PubMed (for more information of the text we use, see acknowledgement). The pre-trained RoBERTa model is post-trained on this corpus by MLM (masked language modeling).

2. task-adaption

   We use domain-specific data from downstream tasks, ChemProt and BioASQ. The post-trained model can be further post-trained on this corpus by MLM (masked language modeling). This idea is proved useful [[2]](#2).

3. vocabulary-expansion

   We train domain-specific tokenizer from PubMed data and merge it with the default tokenizer. The merged tokenizer can be further utilized in finetuning/post-training/task-adaption stage.

## Usage

### Large-scale post-training

We use PubMed data for post-training. To obtain the data:

```bash
tar -zvxf pubmed.tar.gz
```

To save the checkpoints of post-trained models, let's create a directory:

```bash
mkdir models
```

For large scale post-training (on PubMed data): 

```bash
python post_train.py \
	--output_dir models/posttrain-roberta-pubmed 
```

This will save all the checkpoints during training into the directory `models/posttrain-roberta-pubmed`.

### Small-scale task-adaption

After post-training, experiments show that smaller-scale post-training on task data can further improve the performance.

Let's use the post-trained model from last step. Here, we choose `models/posttrain-roberta-pubmed/checkpoint-93000` as an example.

For task-adaption, we need to specify `--post_type adapt`.

For small scale post-training (task-adaption on BioASQ/ChemProt data):

```bash
python post_train.py  \
	--post_type adapt  \
	--load_model_path models/posttrain-roberta-pubmed/checkpoint-93000  \
	--output_dir models/adapttrain-roberta-pubmed
```

### Fine-tuning

After post-training, we can fine-tune our model on various downstream tasks. 

Let's create another directory for the outputs:

```bash
mkdir output
```

For example, we can directly fine-tune the pre-trained RoBERTa model without any post-training displayed above:

```bash
python fine_tune.py  \
	--dataset_name chemprot  \
	--output_dir output/finetune_chemprot
```

We can also fine-tune the post-trained model. Here, we still choose `models/posttrain-roberta-pubmed/checkpoint-93000` as an example.

```bash
python fine_tune.py  \
	--load_model_path models/posttrain-roberta-pubmed/checkpoint-93000  \
	--dataset_name chemprot  \
	--output_dir output/finetune_posttrain_chemprot
```

Moreover, we can use the model after PubMed post-training and task-adaption.

```bash
python fine_tune.py  \
	--load_model_path models/adapttrain-roberta-pubmed/checkpoint-1000  \
	--dataset_name chemprot  \
	--output_dir output/finetune_posttrain_adapttrain_chemprot
```

You can also adjust training arguments such as training epochs `--num_train_epochs` and batch size `--per_device_train_batch_size` . For a detailed view of all training arguments, please see https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py.



### Vocabulary-expansion

#### Create a merged tokenizer

To use an expanded vocabulary, we need to create a specific tokenizer and merge the specific tokenizer into the general tokenizer.

Let's create a directory for tokenizers:

```bash
mkdir tokenizers
```

Then let's train a tokenizer and merge it with roberta's tokenizer.

```bash
python create_tokenizers.py  \
	--merged_tokenizer_output tokenizers/merged_pubmed_roberta
```

You can also specify other arguments such as the vocab size and max added words.

You may get the output information as follows if you specify `--vocab_size 1000 --max_added_words 500`:

```
------Train a specific tokenizer with vocabulary size 1000------
...
------Merge the default Roberta tokenizer with the specific tokenizer------
Max Added Words: 500
[ SAME ] The specific tokenizer introduce 746 same words.
[ DIFF ] The specific tokenizer introduce 254 new words.
[ BEFORE ] tokenizer vocab size: 50265
[ AFTER ] tokenizer vocab size: 50265+254=50519
Save merged tokenizer to tokenizers/merged_pubmed_roberta
```

#### Use the merged tokenizer

You can choose to apply this merged tokenizer in post training, adapt training, or finetuning.

If the vocabulary size of the tokenizer is expanded and the vocabulary size of the model is default, extra word embeddings will be introduced to the model in a random way. You can specify how to initialize the extra word embeddings. Different strategies will be briefly introduced in the report.

For example, add extra words during post training with extra embedding initialized as mean.

```bash
python post_train.py  \
	--output_dir models/posttrain-roberta-pubmed-merged  \
	--use_merged_tokenizer True  \
	--load_tok_path tokenizers/merged_pubmed_roberta
```


## Acknowledgement
The corpus we provide here is a smaller version of the preprocessed PubMed texts provided by https://github.com/ncbi-nlp/bluebert.git. Ours is approximately 130M.

A large part of this project is dependent on huggingface https://github.com/huggingface/transformers.git.

The idea of embedding initialization is partly credited to https://nlp.stanford.edu/~johnhew/vocab-expansion.html

## Reference
<a id="1">[1]</a> 
Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). 
Roberta: A robustly optimized bert pretraining approach. 
arXiv preprint arXiv:1907.11692.

<a id="2">[2]</a>
Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., & Smith, N. A. (2020). 
Don't stop pretraining: adapt language models to domains and tasks. 
arXiv preprint arXiv:2004.10964.