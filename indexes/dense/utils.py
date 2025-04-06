import torch

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(last_hidden_state, attention_mask):
    s = torch.sum(
        last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1
    )
    d = attention_mask.sum(dim=1, keepdim=True).float()
    return s / d

def cls_pooling(last_hidden_state):
    return last_hidden_state[:, 0]