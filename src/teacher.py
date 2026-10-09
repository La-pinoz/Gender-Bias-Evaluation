"""Teacher LLM scoring: one forward pass per prompt, Yes/No read from next-token logits.

Main score = log-odds = logsumexp(logits of the "Yes" tokens) - logsumexp(logits of the "No" tokens),
computed from raw float32 logits. P(Yes) = sigmoid(log-odds). A shortlist decision is
log-odds > 0 (the same as P(Yes) > 0.5). Log-odds are used because P(Yes) saturates near 0
for most bios (Phase 0), which makes probabilities useless for telling bios apart.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_teacher(model_id, device_map=None):
    """Tokenizer with LEFT padding (pad = eos if missing) and an fp16 model on GPU 0, in eval mode.
    Left padding puts the answer position last for every row of a batch."""
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device_map or {"": 0})
    model.eval()
    return tok, model


def answer_token_ids(tok, words):
    """Sorted unique first-token ids of each word variant, e.g. ["Yes", " Yes", "yes", " yes", "YES"]."""
    return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})


@torch.no_grad()
def score_batch(model, tok, prompts, yes_ids, no_ids):
    """Returns a dict of numpy arrays: logodds, p_yes, yes_no_mass.

    logodds     = logsumexp(logits[:, yes_ids]) - logsumexp(logits[:, no_ids])
    p_yes       = sigmoid(logodds)
    yes_no_mass = softmax(logits)[:, yes_ids + no_ids].sum(-1)  (should be ~1: the model answers in format)
    """
    yes_ids, no_ids = list(yes_ids), list(no_ids)
    assert not set(yes_ids) & set(no_ids), "Yes and No token ids overlap"
    # the chat template already adds the special tokens
    enc = tok(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    try:
        out = model(**enc, logits_to_keep=1)   # only the last position's logits
    except TypeError:
        out = model(**enc)
    logits = out.logits[:, -1, :].float()
    logodds = torch.logsumexp(logits[:, yes_ids], dim=-1) - torch.logsumexp(logits[:, no_ids], dim=-1)
    mass = torch.softmax(logits, dim=-1)[:, yes_ids + no_ids].sum(-1)
    return {"logodds": logodds.cpu().numpy(),
            "p_yes": torch.sigmoid(logodds).cpu().numpy(),
            "yes_no_mass": mass.cpu().numpy()}
