from transformers import AutoTokenizer, AutoModel
from utils import cls_pooling, mean_pooling
from datasets import load_dataset
import torch
import faiss
from tqdm import tqdm
from faiss import write_index, read_index

RAW_FILE = "/home/uyen/workspace/nlp_project/data/zac-data/wikipedia_20220620_cleaned_v2.csv"
CACHE_DIR = "/home/uyen/workspace/nlp_project/cache"

# Load model from HuggingFace Hub
# model_name = 'bkai-foundation-models/vietnamese-bi-encoder'
model_name = 'BAAI/bge-m3'
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = CACHE_DIR)
model = AutoModel.from_pretrained(model_name, cache_dir = CACHE_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to('cuda')
model.eval()

TOTAL_CHUNK = 1944407
collection_dataset = load_dataset('csv', data_files=RAW_FILE, streaming=True)["train"]

# dim = 768
dim = 1024
bert_index = faiss.IndexFlatL2(dim)   # build the index

# res = faiss.StandardGpuResources()  # use a single GPU
# index_flat = faiss.IndexFlatL2(dim)  # build a flat (CPU) index
# bert_index = faiss.index_cpu_to_gpu(res, 0, index_flat)

def encode_batch(batch):
    # Tokenize sentences
    encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)

    # Compute token embeddings
    with torch.no_grad():
        last_hidden_state = model(**encoded_input).last_hidden_state

    # Perform pooling. In this case, mean pooling.
    # sentence_embeddings = mean_pooling(last_hidden_state, encoded_input['attention_mask'])
    sentence_embeddings = cls_pooling(last_hidden_state)
    return sentence_embeddings

sentences = []
for item in tqdm(collection_dataset, total = TOTAL_CHUNK):
    text = str(item['text'])
    if text == '':
        text = 'None'

    sentences.append(text)
    if len(sentences) < 128:
        continue

    sentence_embeddings = encode_batch(sentences)
    bert_index.add(sentence_embeddings.detach().cpu().numpy())
    sentences = []

if len(sentences) > 0:
    sentence_embeddings = encode_batch(sentences)
    bert_index.add(sentence_embeddings.detach().cpu().numpy())

# bert_index = faiss.index_gpu_to_cpu(bert_index)
write_index(bert_index, "/home/uyen/workspace/nlp_project/data/cache/dense_index/bge-m3.index")
# index = read_index("large.index")
