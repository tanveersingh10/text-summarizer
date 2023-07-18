import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self,example_batch):
        # The tokenizer returns a dictionary containing the following keys:
        # 'input_ids': a list of token ids to be fed to a model
        # 'attention_mask': a list of integers specifying which tokens should be attended to by the model
        input_encodings = self.tokenizer(text=example_batch['dialogue'], max_length=1024, truncation=True)
        
        target_encodings = self.tokenizer(text_target=example_batch['summary'], max_length=128, truncation=True,
    )   
        return {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
        # Had to google what attention_mask was so writing it here so I remember 
        # An attention mask is a binary tensor indicating the position of the padded indices so that the model does not attend to them.

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path) 
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched = True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir,"samsum_dataset")) 