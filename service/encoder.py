from transformers import AutoTokenizer, AutoModel
from .utils import mean_pooling
import torch


class Encoder:

    def __init__(self, model_name: str, cache_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def encode(self, sentences: list):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=256, return_tensors='pt').to(self.device)
        
        # Compute token embeddings
        with torch.no_grad():
            last_hidden_state = self.model(**encoded_input).last_hidden_state

        sentence_embeddings = mean_pooling(last_hidden_state, encoded_input['attention_mask'])
        sentence_embeddings = sentence_embeddings.detach().cpu().numpy()

        return sentence_embeddings