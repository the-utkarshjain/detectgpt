import torch
import torch.nn.functional as F

# get average entropy of each token in the text
def get_entropy(args, config, text):
    assert args.openai_model is None, "get_entropy not implemented for OpenAI models"

    DEVICE = args.DEVICE
    base_model = config["base_model"]
    base_tokenizer = config["base_tokenizer"]

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()