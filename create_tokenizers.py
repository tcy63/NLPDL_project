from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.models import (
    BPE,
    WordPiece
)
from tokenizers.normalizers import (
    Sequence,
    NFD, 
    Lowercase, 
    StripAccents
)
from tokenizers.trainers import (
    BpeTrainer,
    WordPieceTrainer
)
from tokenizers.pre_tokenizers import (
    ByteLevel
)
from transformers import (
    HfArgumentParser,
    RobertaTokenizerFast,
    AutoTokenizer
)
from dataclasses import (
    dataclass, 
    field
)
import sys
import os

@dataclass
class TokenizerArguments:
    do_train: bool=field(
        default=True,
        metadata={"help": "whether to train the specific toeknizer."}
    )
    data_file: str=field(
        default="pubmed_base.txt",
        metadata={"help": "file or files used to train the tokenizer."}
    )
    tok_type: str=field(
        default="wordpiece",
        metadata={"help": "the type of tokenizer."}
    )
    specific_tokenizer_output: str=field(
        default=None,
        metadata={"help": "the json file path to save the trained specific toeknizer, used by 'tokenizer.save'. The basename without extension will be used as a directory to save the wrapped tokenizer by 'tokenizer.save_pretrained'."}
    )
    specific_tokenizer_input: str=field(
        default=None,
        metadata={"help": "the json file to load the trained specific toeknizer."}
    )
    vocab_size: int=field(
        default=30000,
        metadata={"help": "the size of the vocab."}
    )
    do_merge: bool=field(
        default=True,
        metadata={"help": "whether to merge the specific toeknizer with default roberta tokenizer."}
    )
    max_added_words: int=field(
        default=500,
        metadata={"help": "if merge, the maximum number of specific words added into the default roberta vocabulary."}
    )
    merged_tokenizer_output: str=field(
        default=None,
        metadata={"help": "the directory path to save the merged toeknizer."}
    )
    wrap: bool=field(
        default=False,
        metadata={"help":"whether to save the wrapped specific tokenizer."}
    )
        


def train_tokenizer(file, tok_type, vocab_size=30000):
    files = []
    files.append(file)
    Models = {
        "wordpiece": WordPiece,
        "bpe": BPE
    }
    Trainers = {
        "wordpiece": WordPieceTrainer,
        "bpe": BpeTrainer
    }
    tokenizer = Tokenizer(Models[tok_type](unk_token="<unk"))
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    special_tokens = ["<unk>","<pad>","<mask>","<s>","</s>"]
    trainer = Trainers[tok_type](vocab_size=vocab_size, special_tokens=special_tokens, continuing_subword_prefix='Ġ')
    tokenizer.model = Models[tok_type](unk_token="<unk>")
    tokenizer.train(files, trainer=trainer)
    return tokenizer
        
        
        
def merge_tokenizer(general_tokenizer, specific_tokenizer, max_added_words):
    specific_vocab = [k for k, v in specific_tokenizer.get_vocab().items()]
    specific_size = len(specific_vocab)
    general_vocab = [k for k, v in general_tokenizer.get_vocab().items()]
    general_size = len(general_vocab)
    
    same_tokens_list = []
    diff_tokens_list = []
    
    # loop over each word in specific_vocab
    for idx_new, w in enumerate(specific_vocab):
        try:
            idx_old = general_vocab.index(w)
        except:
            idx_old = -1
        if idx_old >= 0:
            same_tokens_list.append((w, idx_new))
        else:
            diff_tokens_list.append((w, idx_new))
    
    new_tokens = [k for k, v in diff_tokens_list]
      
    print(f"[ SAME ] The specific tokenizer introduce {len(same_tokens_list)} same words.")
    print(f"[ DIFF ] The specific tokenizer introduce {len(new_tokens)} new words.")
    

    if len(new_tokens) > max_added_words:
        new_tokens = new_tokens[:max_added_words]

    added_tokens = general_tokenizer.add_tokens(new_tokens)
    
    print("[ BEFORE ] tokenizer vocab size:", general_size) 
    print("[ AFTER ] tokenizer vocab size: {}+{}={}".format(general_size,added_tokens, general_size+added_tokens)) 

    return general_tokenizer, added_tokens

def main():
    parser = HfArgumentParser([TokenizerArguments])
    tok_args = parser.parse_args_into_dataclasses()[0]

    '''
    train a new tokenizer on specific corpus
    '''
    if tok_args.do_train:
        print(f"------Train a specific tokenizer with vocabulary size {tok_args.vocab_size}------")
        tokenizer = train_tokenizer(tok_args.data_file, tok_args.tok_type, tok_args.vocab_size)
        if tok_args.specific_tokenizer_output != None:
            print(f"------Save specific tokenizer to {tok_args.specific_tokenizer_output}------")
            tokenizer.save(tok_args.specific_tokenizer_output) # in json format
            if tok_args.wrap:
                print(f"------Save the wrapped specific tokenizer to {tok_args.specific_tokenizer_output.split('.')[0]} for future loading from pretrained------")
                wrapped_tokenizer = RobertaTokenizerFast(tokenizer_object=tokenizer)
                wrapped_tokenizer.save_pretrained(tok_args.specific_tokenizer_output.split('.')[0]) # a directory
                
    else:
        print(f"------Load specific tokenizer from {tok_args.specific_tokenizer_input}------")
        tokenizer = Tokenizer.from_file(tok_args.specific_tokenizer_input)
    
    '''
    merge a specific tokenizer with the default roberta-base tokenizer
    '''
    if tok_args.do_merge:
        roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        print(f"------Merge the default Roberta tokenizer with the specific tokenizer------")
        print(f"Max Added Words: {tok_args.max_added_words}")
        tokenizer, add_tokens = merge_tokenizer(roberta_tokenizer, tokenizer, tok_args.max_added_words)
        if tok_args.merged_tokenizer_output != None:
            print("Save merged tokenizer to {}".format(tok_args.merged_tokenizer_output))
            tokenizer.save_pretrained(tok_args.merged_tokenizer_output)
            
if __name__ == "__main__":
    main()
            