"""Student distillation: input text, paired dataset, collate, consistency loss, training, prediction.

Targets are STANDARDISED teacher log-odds: t = (clip(logodds, lo, hi) - mean) / sd, with lo, hi, mean, sd
computed on the TRAIN split only. lambda is relative to these standardised units.
"""
import copy, random, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)


def make_input(job_title, name, bio):
    """(text_a, text_b) for every student and baseline. The name is included because the teacher saw it."""
    return job_title, f"Candidate: {name}. {bio.strip()}"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class PairedDataset(Dataset):
    """One example per item. `pairs` has columns: text_a, text_b_m, text_b_f, t_m, t_f (standardised).
    In CDA mode the caller passes t_m = t_f = (t_m + t_f) / 2 before building the dataset."""
    def __init__(self, pairs):
        self.a = pairs.text_a.tolist(); self.bm = pairs.text_b_m.tolist(); self.bf = pairs.text_b_f.tolist()
        self.tm = pairs.t_m.to_numpy(np.float32); self.tf = pairs.t_f.to_numpy(np.float32)
    def __len__(self):
        return len(self.a)
    def __getitem__(self, i):
        return self.a[i], self.bm[i], self.bf[i], self.tm[i], self.tf[i]


def make_collate(tok, max_length):
    def collate(batch):
        a, bm, bf, tm, tf = zip(*batch)
        kw = dict(truncation="only_second", max_length=max_length, padding=True, return_tensors="pt")
        return {"enc_m": tok(list(a), list(bm), **kw), "enc_f": tok(list(a), list(bf), **kw),
                "t_m": torch.tensor(tm), "t_f": torch.tensor(tf)}
    return collate


def paired_loss(p_m, p_f, t_m, t_f, lam):
    """loss = 0.5*(MSE(p_m,t_m) + MSE(p_f,t_f)) + lam * mean((p_m - p_f)^2). All float32."""
    mse = 0.5 * (torch.mean((p_m - t_m) ** 2) + torch.mean((p_f - t_f) ** 2))
    cons = torch.mean((p_m - p_f) ** 2)
    return mse + lam * cons, mse, cons


def load_student(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1)
    return tok, model


@torch.no_grad()
def predict(model, tok, text_a, text_b, max_length, batch_size, device):
    """Standardised predictions (numpy float64) for parallel lists text_a / text_b, in input order."""
    model.eval()
    text_a, text_b = list(text_a), list(text_b)
    use_cuda = str(device).startswith("cuda")
    out = []
    for i in range(0, len(text_a), batch_size):
        enc = tok(text_a[i:i + batch_size], text_b[i:i + batch_size], truncation="only_second",
                  max_length=max_length, padding=True, return_tensors="pt").to(device)
        if use_cuda:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**enc).logits
        else:
            logits = model(**enc).logits
        out.append(logits.squeeze(-1).float().cpu())
    return torch.cat(out).numpy().astype(np.float64)


def train_student(train_pairs, val_pairs, cfg, lam, seed, device, log=print):
    """Train one student. Returns (model with BEST-epoch weights loaded, tok, history, best_epoch).
    history = list of dicts per epoch: epoch, train_loss, val_loss, val_mse, val_cons, seconds.
    Best epoch = lowest val_loss (same formula as training, same lam).
    The model weights stay in float32; only the forward pass runs under fp16 autocast."""
    set_seed(seed)
    tok, model = load_student(cfg["student_model_id"])
    model.to(device)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(PairedDataset(train_pairs), batch_size=cfg["batch_pairs"], shuffle=True,
                        collate_fn=make_collate(tok, cfg["max_length"]), num_workers=2, generator=g)
    no_decay = ["bias", "LayerNorm.weight"]
    params = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": cfg["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
               "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=cfg["lr"])
    total = len(loader) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total), total)
    scaler = torch.amp.GradScaler("cuda")
    best, best_epoch, history = None, -1, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        run = []
        for batch in loader:
            enc_m = {k: v.to(device) for k, v in batch["enc_m"].items()}
            enc_f = {k: v.to(device) for k, v in batch["enc_f"].items()}
            t_m, t_f = batch["t_m"].to(device), batch["t_f"].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                p_m = model(**enc_m).logits.squeeze(-1)
                p_f = model(**enc_f).logits.squeeze(-1)
            loss, mse, cons = paired_loss(p_m.float(), p_f.float(), t_m, t_f, lam)   # loss in float32
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(loss.item())
        # validation: both versions of val_pairs, same loss formula and lam as training
        vm = predict(model, tok, val_pairs.text_a.tolist(), val_pairs.text_b_m.tolist(),
                     cfg["max_length"], cfg["eval_batch"], device)
        vf = predict(model, tok, val_pairs.text_a.tolist(), val_pairs.text_b_f.tolist(),
                     cfg["max_length"], cfg["eval_batch"], device)
        vl, vmse, vcons = (x.item() for x in paired_loss(
            torch.tensor(vm), torch.tensor(vf),
            torch.tensor(val_pairs.t_m.values), torch.tensor(val_pairs.t_f.values), lam))
        history.append(dict(epoch=epoch, train_loss=float(np.mean(run)), val_loss=vl, val_mse=vmse,
                            val_cons=vcons, seconds=time.time() - t0))
        log(f"    epoch {epoch}: train {np.mean(run):.4f} | val {vl:.4f} (mse {vmse:.4f}, cons {vcons:.4f})")
        if best is None or vl < history[best_epoch - 1]["val_loss"]:
            best = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            best_epoch = epoch
    model.load_state_dict(best)
    return model, tok, history, best_epoch
