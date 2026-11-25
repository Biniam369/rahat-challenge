import json, re, copy
from invoice_ml.text_utils import parse_number, to_western_digits

def flatten(obj, path=""):
    """{'a': {'b': 1}} -> {'a.b': 1}"""
    if isinstance(obj, dict):
        return {k2: v2 for k, v in obj.items() if not k.startswith("_")
                for k2, v2 in flatten(v, f"{path}.{k}" if path else k).items()}
    if isinstance(obj, list):
        return {k2: v2 for i, v in enumerate(obj) for k2, v2 in flatten(v, f"{path}[{i}]").items()}
    return {path: obj}

def same(a, b):
    """Numbers compared as numbers (so ٢٤ == 24, 1,500 == 1500); text compared after trimming."""
    na, nb = parse_number(a), parse_number(b)
    if na is not None and nb is not None and re.fullmatch(r"[\d.,()\s٠-٩٫٬-]+", str(a)):
        return abs(na - nb) < 0.005
    return str(a or "").strip() == str(b or "").strip()

def cer(pred, truth):
    """Character error rate = edit distance / length of truth."""
    p, t = to_western_digits(str(pred or "")), to_western_digits(str(truth or ""))
    prev = list(range(len(t) + 1))
    for i, cp in enumerate(p, 1):
        cur = [i]
        for j, ct in enumerate(t, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cp != ct)))
        prev = cur
    return prev[-1] / max(len(t), 1)

CRITICAL = re.compile(r"qty|unit_price|amount|subtotal|total|vat|discount|trn|date")

def extraction_accuracy(pred: dict, truth: dict):
    P, T = flatten(pred), flatten(truth)
    rows = [{"field": f, "truth": T[f], "pred": P.get(f), "correct": same(P.get(f), T[f]),
             "critical": bool(CRITICAL.search(f)), "cer": cer(P.get(f), T[f])} for f in T]
    n = len(rows); ok = sum(r["correct"] for r in rows)
    crit = [r for r in rows if r["critical"]]
    print(f"Field accuracy:     {ok}/{n} = {ok/n:.1%}")
    print(f"Critical accuracy:  {sum(r['correct'] for r in crit)}/{len(crit)} = "
          f"{sum(r['correct'] for r in crit)/len(crit):.1%}   (amounts, qty, VAT, TRN, dates)")
    print(f"Mean CER (text):    {sum(r['cer'] for r in rows)/n:.2%}")
    print(f"Missing fields: {len(set(T) - set(P))} | Extra fields: {len(set(P) - set(T))}")
    for r in rows:
        if not r["correct"]:
            print(f"  WRONG {r['field']}: extracted={r['pred']!r} truth={r['truth']!r}")
    return rows

if __name__ == "__main__":
    pred = json.load(open("data/invoice_extracted.json", encoding="utf-8"))
    # Demo ground truth: a copy with 2 deliberate differences (in real use, a person types this file)
    truth = copy.deepcopy(pred)
    truth["totals_printed"]["vat_amount"] = "1,123.74"
    truth["sections"][0]["items"][2]["description_ar"] = "رمل مغسول ناعم"
    extraction_accuracy(pred, truth)
