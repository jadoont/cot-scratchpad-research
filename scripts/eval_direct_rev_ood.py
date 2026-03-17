import re
import pickle
from pathlib import Path
from collections import defaultdict
import math
import torch

from model import GPTConfig, GPT

def load_char_meta(dataset_dir: Path):
    meta = pickle.loads((dataset_dir / "meta.pkl").read_bytes())
    stoi = meta["stoi"]
    itos = meta["itos"]
    encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
    decode = lambda ids: "".join(itos[int(i)] for i in ids)
    return encode, decode

def load_model(out_dir: Path, device: str):
    ckpt = torch.load(out_dir / "ckpt.pt", map_location=device)
    config = GPTConfig(**ckpt["model_args"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def generate(model, x, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, _ = model(x)
        logits = logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        x = torch.cat((x, next_id), dim=1)
        if x.size(1) >= model.config.block_size:
            break
    return x

def count_carries(a, b):
    carries = 0
    carry = 0
    while a > 0 or b > 0:
        da = a % 10
        db = b % 10
        s = da + db + carry
        if s >= 10:
            carries += 1
            carry = 1
        else:
            carry = 0
        a //= 10
        b //= 10
    return carries

def parse_pred(text):
    m = re.search(r"=(\d+)", text)
    return m.group(1) if m else None

def unrev_num_str(s):
    return str(int(s[::-1])) if s else None

def eval_file(model, encode, decode, path, device="mps", n_eval=500):
    lines = path.read_text(encoding="utf-8").splitlines()[:n_eval]
    stats = defaultdict(lambda: {"n": 0, "correct": 0})
    examples = []

    for ln in lines:
        m = re.match(r"^(\d+)\+(\d+)=(\d+)$", ln.strip())
        if not m:
            continue
        a_rev, b_rev, gold_rev = m.group(1), m.group(2), m.group(3)

        a = int(a_rev[::-1])
        b = int(b_rev[::-1])
        gold = int(gold_rev[::-1])

        prompt = f"{a_rev}+{b_rev}="
        x = encode(prompt).unsqueeze(0).to(device)
        max_new = min(8, model.config.block_size - x.size(1))
        y = generate(model, x, max_new_tokens=max_new)
        gen = decode(y[0].tolist())

        pred_rev = parse_pred(gen)
        pred = int(pred_rev[::-1]) if pred_rev is not None else None

        ok = (pred == gold)

        digits = max(len(str(a)), len(str(b)))
        carries = count_carries(a, b)

        stats[f"digits={digits}"]["n"] += 1
        stats[f"digits={digits}"]["correct"] += int(ok)
        stats[f"carries={carries}"]["n"] += 1
        stats[f"carries={carries}"]["correct"] += int(ok)

        if not ok and len(examples) < 8:
            examples.append((prompt, gold, pred, gen))

    return stats, examples

def print_stats(title, stats, examples):
    print(f"\n=== {title} ===")
    for k in sorted(k for k in stats if k.startswith("digits=")):
        d = stats[k]
        acc = d["correct"] / d["n"] if d["n"] else math.nan
        print(f"{k:10s} acc={acc:.3f}  n={d['n']}")
    for k in sorted(k for k in stats if k.startswith("carries=")):
        d = stats[k]
        acc = d["correct"] / d["n"] if d["n"] else math.nan
        print(f"{k:10s} acc={acc:.3f}  n={d['n']}")
    print("\nSample errors:")
    for p, g, pr, gen in examples:
        print(f"{p} gold={g} pred={pr} gen={gen[:80]}")

if __name__ == "__main__":
    DEVICE = "mps"
    dataset_dir = Path("data/addition_rev/direct")
    out_dir = Path("out/addition_rev_direct_b128")

    encode, decode = load_char_meta(dataset_dir)
    model = load_model(out_dir, DEVICE)

    stats2, ex2 = eval_file(model, encode, decode, Path("data/addition_rev/test2_direct.txt"), device=DEVICE)
    stats3, ex3 = eval_file(model, encode, decode, Path("data/addition_rev/test3_direct.txt"), device=DEVICE)

    print_stats("REVERSED DIRECT TEST 2-DIGIT", stats2, ex2)
    print_stats("REVERSED DIRECT TEST 3-DIGIT", stats3, ex3)
