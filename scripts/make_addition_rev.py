import random
from pathlib import Path

random.seed(0)

OUTDIR = Path("data/addition_rev")

TRAIN_N = 50000
VAL_N = 5000
TEST_N = 3000

TRAIN_DIGITS = [2]
VAL_DIGITS = [2]
TEST2_DIGITS = [2]
TEST3_DIGITS = [3]

def sample_pair(ndigits):
    lo = 10 ** (ndigits - 1)
    hi = 10 ** ndigits - 1
    return random.randint(lo, hi), random.randint(lo, hi)

def revnum(n: int) -> str:
    return str(n)[::-1]

def cot_steps(a, b):
    carry = 0
    aa, bb = a, b
    i = 0
    out = []
    while aa > 0 or bb > 0 or i == 0:
        da = aa % 10
        db = bb % 10
        s = da + db + carry
        digit = s % 10
        new_carry = 1 if s >= 10 else 0
        out.append(f"d{i}:{da}+{db}+{carry}={s}->{digit},c{new_carry}")
        carry = new_carry
        aa //= 10
        bb //= 10
        i += 1
    return ";".join(out)

def fmt_example(a, b, mode):
    ans = a + b

    # reverse inputs and outputs
    ra = revnum(a)
    rb = revnum(b)
    rans = revnum(ans)

    prompt = f"{ra}+{rb}="

    if mode == "direct":
        return f"{prompt}{rans}\n"
    if mode == "cot":
        return f"{prompt}{cot_steps(a, b)}|{rans}\n"
    if mode == "scratch":
        return f"{prompt}<scratch>{cot_steps(a, b)}</scratch>{rans}\n"
    raise ValueError(mode)

def write_file(path, n, digits, mode):
    with path.open("w", encoding="utf-8") as f:
        for _ in range(n):
            nd = random.choice(digits)
            a, b = sample_pair(nd)
            f.write(fmt_example(a, b, mode))

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    for mode in ["direct", "cot", "scratch"]:
        write_file(OUTDIR / f"train_{mode}.txt", TRAIN_N, TRAIN_DIGITS, mode)
        write_file(OUTDIR / f"val_{mode}.txt", VAL_N, VAL_DIGITS, mode)

    write_file(OUTDIR / "test2_direct.txt", TEST_N, TEST2_DIGITS, "direct")
    write_file(OUTDIR / "test3_direct.txt", TEST_N, TEST3_DIGITS, "direct")

    write_file(OUTDIR / "test2_cot.txt", TEST_N, TEST2_DIGITS, "cot")
    write_file(OUTDIR / "test3_cot.txt", TEST_N, TEST3_DIGITS, "cot")

    write_file(OUTDIR / "test2_scratch.txt", TEST_N, TEST2_DIGITS, "scratch")
    write_file(OUTDIR / "test3_scratch.txt", TEST_N, TEST3_DIGITS, "scratch")

    print("Wrote reversed datasets to", OUTDIR)

if __name__ == "__main__":
    main()
