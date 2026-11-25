"""Generate track_a_colab.ipynb from the repo files, so notebook and repo never drift.

Usage: python tools/build_notebook.py
Each source file is embedded as a `%%writefile` cell; running the notebook in
Colab recreates the repo under /content/jubii7-track-a and runs every task.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n")})


def writefile(rel: str) -> None:
    body = (ROOT / rel).read_text(encoding="utf-8")
    code(f"%%writefile {rel}\n{body}")


# ------------------------------------------------------------------ intro
md("""
# jubii.7 · Track A: "Read the messy invoice"

One scanned page (a bilingual Arabic/English tax invoice and delivery note from **Al Waha Building
Materials**) goes in. Out come clean JSON, a list of every problem on the page, and a model that
puts each line item into **cement / steel / aggregates / blocks / services**, served as a `/predict` API.

| Task | Where |
|---|---|
| 1 Data understanding: extract, profile, recalculate, list problems | §1 |
| 2 Labelling: 5-line guide + AI-made lines, checked and fixed | §2 |
| 3 Model: TF-IDF + logistic regression | §3 |
| 4 MLflow: 3 runs compared, best one picked | §4 |
| 5 Evaluation: per-class P/R/F1, confusion matrix, mistakes | §5 |
| 6 Deployment: FastAPI `/predict`, live call | §6 |
| 7 Code & docs: README, known limits | §7 |

**Run all** (Runtime → Run all). Each `%%writefile` cell writes one file of the repo, so the notebook
*is* the repo: download it as a zip at the end.
""")

code("""
import os, sys, subprocess, json, time, warnings
from pathlib import Path
import requests

warnings.filterwarnings("ignore")              # keep library warnings out of the outputs
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mlflow", "fastapi", "uvicorn"], check=True)

if IN_COLAB:
    PROJECT = Path("/content/jubii7-track-a")
elif (Path.cwd() / "invoice_ml").exists():          # opened inside the cloned repo
    PROJECT = Path.cwd()
else:
    PROJECT = Path.cwd() / "jubii7-track-a"
for d in ["invoice_ml", "app", "data", "docs", "reports", "models"]:
    (PROJECT / d).mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT)
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONWARNINGS"] = "ignore"


def run(module, quiet=False):
    \"\"\"Run `python -m <module>` (exactly what the README says) and show its output.\"\"\"
    r = subprocess.run([sys.executable, "-m", module], capture_output=True, text=True, encoding="utf-8")
    if not quiet:
        print(r.stdout)
    if r.returncode:
        print(r.stderr[-3000:]); raise RuntimeError(f"{module} failed")


print("Running in Colab:", IN_COLAB, "| project folder:", PROJECT.name)
""")

writefile("invoice_ml/__init__.py")

md("""
### Shared helper: normalise Arabic text and digits
The page mixes `٢٤` with `24` and `١٨` with `18`, and writes amounts as `(155.20)` or `1,500`.
Every number goes through `parse_number`, and every description goes through `normalize_text`
(which also unifies alef/taa-marbuta and drops diacritics) before the model sees it.
The amount in words is Arabic, so `arabic_words_to_number` turns it back into a figure.
""")
writefile("invoice_ml/text_utils.py")

code("""
from invoice_ml.text_utils import parse_number, arabic_words_to_number, normalize_text
print(parse_number("٢٤"), parse_number("1,500"), parse_number("(155.20)"), parse_number("١٫٨"))
print(arabic_words_to_number("فقط ثلاثة وعشرون ألفاً وستمائة وستون درهماً وثلاثون فلساً لا غير"))
print(normalize_text("أسمنت بورتلاند ٥٠ كجم"))
""")

# ------------------------------------------------------------------ task 1
md("""
## 1 · Data understanding

**Extraction.** The scan is transcribed to JSON (vision LLM + the prompt in `docs/PROMPTS.md`, then
checked by eye) with every value **exactly as printed**: Arabic digits, commas, brackets, blanks.
Fixing values during extraction would hide the very errors the finance team needs to catch.
Uncertain reads go in `_low_confidence`.
""")
writefile("data/invoice_extracted.json")

md("""
**Checker.** It profiles every field (type and digit script), then recalculates line by line:
amount = delivered qty × price, then section subtotals, total before discount, the 2% Site B discount,
taxable amount, 5% VAT, grand total and the amount in words. Each total is compared **two ways**:
- against the *printed* figures above it, which shows where the page's own arithmetic broke
- against the *full recalculation*, which gives what the number should be

It also cross-checks the delivery table (dates, receivers, trips), English vs Arabic terms, the payment
boxes, TRNs, signatures and the stamp.
""")
writefile("invoice_ml/check_invoice.py")

code("""
import pandas as pd
pd.set_option("display.max_colwidth", 120)
from invoice_ml.check_invoice import profile, check

inv = json.loads(Path("data/invoice_extracted.json").read_text(encoding="utf-8"))
prof = profile(inv)
print("Field types:", prof["type"].value_counts().to_dict())
print("Digit scripts:", prof["digits"].value_counts().to_dict())
display(prof[prof["digits"].isin(["arabic", "mixed"]) | (prof["type"] == "missing")])
""")

code("""
clean, problems = check(inv)
totals = pd.DataFrame({"printed": clean["totals_printed"], "recalculated": clean["totals_recalculated"]})
totals["difference"] = totals["printed"] - totals["recalculated"]
display(totals.style.format("{:,.2f}"))
print(f"{len(problems)} problems:", problems["severity"].value_counts().to_dict())
display(problems)
""")

code("""
# Save the clean JSON + reports (same as: python -m invoice_ml.check_invoice)
run("invoice_ml.check_invoice", quiet=True)
print(json.dumps(clean["sections"][0]["items"][2], ensure_ascii=False, indent=2))
""")

md("""
**What the page gets wrong (headline):** the customer is charged **AED 23,606.30** instead of **AED 23,504.04**,
an overcharge of **AED 102.26** caused by two independent errors (1 and 2 below).
1. Line 3, washed sand: 24 × 65 = **1,560**, printed **1,650** (digit swap). It flows into the Site A
   subtotal, total before discount, taxable amount and grand total.
2. VAT printed **1,131.50**, but 5% of the printed taxable amount (22,474.80) is 1,123.74. This is a second, independent error.
3. Amount in words says **23,660.30**, the figure says **23,606.30**, and the correct total is **23,504.04**.
4. Delivery D3 is dated **31/11/2026**, which doesn't exist. D2 (15/11) is after the invoice date (14/11) yet marked received.
5. English terms say **45 days**, Arabic terms say **30 days**.
6. Both **Cash** and **Credit** are ticked. The membrane is billed for 18 of 20 ordered with 2 still pending.
   There are 3 trips but 1 delivery charge, the address is in Ajman while prices cover "Sharjah only", and every signature is blank.
7. Open question for finance: the 2% Site B discount is also applied to the delivery charge. Is that intended?

The full list of all problems, with severity and printed vs expected values, is the table above.
""")

# ------------------------------------------------------------------ task 2
md("""
## 2 · Data labelling

**Labelling guide (5 rules)**
1. Label what is being bought, not words in the name. Ignore sizes, units, brands.
2. **cement**: cement and products made mainly of it (ready-mix, mortar, cement adhesive, additives).
3. **blocks**: any finished unit laid by hand (concrete/hollow/AAC blocks, bricks, kerbs). "Concrete block" is **blocks**.
4. **steel**: steel products only. **services**: labour, transport, hire, fees, *even if steel/concrete is mentioned*.
5. **aggregates**: sand, gravel, crushed stone, whatever the unit. If a line fits no class, label it
   `out_of_scope`. Never force it.

The 6 invoice lines plus 63 AI-generated lines (English and Arabic) were labelled. Every AI label was checked against the
guide: `ai_label` is what the AI said, `label` is the corrected one, and `review_note` says why.
""")
writefile("data/line_items_labelled.csv")

code("""
lab = pd.read_csv("data/line_items_labelled.csv")
print(len(lab), "lines | by source:", lab["source"].value_counts().to_dict(), "| by language:", lab["lang"].value_counts().to_dict())
print("Label counts:", lab["label"].value_counts().to_dict())
fixed = lab[lab["ai_label"] != lab["label"]]
print(f"\\nAI labels corrected: {len(fixed)} of {len(lab)} ({len(fixed)/len(lab):.0%})")
display(fixed[["text", "ai_label", "label", "review_note"]])
""")

md("""
**Pattern in the AI's mistakes:** it labels by keyword (*concrete*, *truck*, *خرسانة*) rather than by
the item, which is exactly what rules 1, 3 and 4 address. The **waterproofing membrane** on the real invoice fits
none of the 5 classes, so it is excluded from training and flagged. The fix is a new class (e.g. `finishing/other`).
""")

# ------------------------------------------------------------------ task 3
md("""
## 3 · Model training: TF-IDF + logistic regression

Why this model:
- **Data size.** About 70 lines is far too few to fine-tune a transformer. A linear model on n-grams fits and trains in milliseconds.
- **Explainable.** The coefficients show which words push a line to a class, which finance can audit.
- **Calibrated-ish probabilities.** Low-confidence lines are routed to a human (`needs_review` in the API).
- **Bilingual for free.** TF-IDF does not care about the script. `normalize_text` makes `٥٠ كجم` = `50 كجم`.

Three feature variants are compared: word 1-2 grams, character 2-5 grams (robust to Arabic prefixes
like ال/بال and typos), and both combined. `class_weight="balanced"` handles the small class differences.
""")
writefile("invoice_ml/model.py")

# ------------------------------------------------------------------ task 4
md("""
## 4 · MLflow tracking

Each variant is one MLflow run. It logs parameters, 5-fold **CV macro-F1** (mean ± std) on the training split,
test macro-F1/accuracy, and the model itself. **Selection uses CV on the training split only**, so the test set stays an
untouched final check. With ties the simpler model wins. The selected run is tagged `selected=true`.
""")
writefile("invoice_ml/train.py")

code("""
run("invoice_ml.train")
""")

code("""
import mlflow
mlflow.set_tracking_uri(f"sqlite:///{(PROJECT / 'mlflow.db').as_posix()}")
runs = mlflow.search_runs(experiment_names=["invoice-line-category"], order_by=["attributes.start_time DESC"])
cols = ["tags.mlflow.runName", "params.features", "params.C", "metrics.cv_macro_f1", "metrics.cv_macro_f1_std",
        "metrics.test_macro_f1", "metrics.test_accuracy", "tags.selected", "run_id"]
display(runs[[c for c in cols if c in runs.columns]].head(3))
""")

code("""
# Chart of the 3 runs (always works, good for a screenshot)
import matplotlib.pyplot as plt
r = runs.head(3).set_index("tags.mlflow.runName").sort_values("metrics.cv_macro_f1")  # latest training
ax = r[["metrics.cv_macro_f1", "metrics.test_macro_f1"]].plot.barh(
    xerr={"metrics.cv_macro_f1": r["metrics.cv_macro_f1_std"]}, figsize=(7, 3))
ax.set_xlim(0, 1); ax.set_xlabel("macro-F1"); ax.set_ylabel("")
ax.set_title("MLflow runs: CV macro-F1 (± std) vs test macro-F1"); ax.legend(["CV (selection)", "test"], loc="lower right")
plt.tight_layout(); plt.show()
""")

md("""
### Open the real MLflow UI
Run the next cell, then click the link. In the UI: **Model training → Runs**, tick all 3 runs → **Compare**
to show params, metrics and the `selected=true` tag side by side.
`--allowed-hosts` lists Colab's proxy domains: MLflow 3 rejects every host except localhost ("Invalid Host header"),
and Colab serves the UI through `*.googleusercontent.com` / `*.colab.dev`.
""")

code("""
mlflow_ui = subprocess.Popen(
    [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", f"sqlite:///{(PROJECT / 'mlflow.db').as_posix()}",
     "--host", "0.0.0.0", "--port", "5000",
     "--allowed-hosts", "*.googleusercontent.com,*.colab.dev,localhost,localhost:*,127.0.0.1,127.0.0.1:*",
     "--cors-allowed-origins", "https://*.googleusercontent.com,https://*.colab.dev"],
    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
for _ in range(40):  # wait until the server answers
    try:
        requests.get("http://127.0.0.1:5000/health", timeout=1); break
    except Exception:
        time.sleep(1)
exp_id = mlflow.get_experiment_by_name("invoice-line-category").experiment_id
runs_path = f"/#/experiments/{exp_id}/runs?workflowType=machine_learning"
if IN_COLAB:
    from google.colab import output
    output.serve_kernel_port_as_window(5000, path=runs_path, anchor_text="Open MLflow UI (runs)")
else:
    print("Open: http://127.0.0.1:5000" + runs_path)
""")

md("""
**Choice.** The CV gap between variants is smaller than one standard deviation (~0.13-0.18). With so
little data, a single held-out test of 17 lines is too noisy to pick a winner. I pick on CV, keep the simplest model, and
say plainly that **more labelled data matters more than the feature choice.**
""")

# ------------------------------------------------------------------ task 5
md("""
## 5 · Evaluation

Two views, because the test split is only 17 lines (one error = 6 points):
- **held-out test set**: the honest final number
- **5-fold out-of-fold predictions over all lines**: more stable, and gives enough errors to study
""")
writefile("invoice_ml/evaluate.py")

code("""
run("invoice_ml.evaluate")
""")

code("""
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from invoice_ml.model import LABELS, SEED, build_pipeline, load_data

best = json.loads(Path("models/best_run.json").read_text())
df = load_data()
oof = cross_val_predict(build_pipeline(best["cfg"]), df["text"], df["label"],
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
ConfusionMatrixDisplay.from_predictions(df["label"], oof, labels=LABELS, cmap="Blues")
plt.title(f"Out-of-fold confusion matrix - {best['name']}"); plt.xticks(rotation=30); plt.show()
""")

md("""
**Three mistakes and why**
1. *Steel shuttering plate rental*: true **services**, predicted **steel** (0.79, the most confident error).
   "steel" is a strong steel word and "rental" appears only once in training. The model learns keywords, the
   same trap the AI labeller fell into. Fix: more hire/rental examples, or a rule "rental/hire → services".
2. *خرسانة جاهزة متر مكعب* (ready-mix concrete): true **cement**, predicted **services**. "خرسانة" shows up in
   "ايجار مضخة خرسانة" (pump hire = services), so the word is split across two classes. With so few examples
   the per-unit word "متر مكعب" isn't enough. Fix: more ready-mix lines in both languages.
3. *Kerb stone 50cm*: true **blocks**, predicted **aggregates**. "stone" usually means crushed stone (aggregates).
   This is a genuine ambiguity that the guide (rule 3) resolves but the data barely covers.

Common thread: keyword overlap between classes and too few examples. Low-confidence predictions (<0.5) are
mostly wrong, which supports routing them to a human.
""")

# ------------------------------------------------------------------ task 6
md("""
## 6 · Deployment: FastAPI `/predict`
Loads the saved pipeline and returns the category, confidence, all class scores and a `needs_review` flag (confidence < 0.5).
""")
writefile("app/main.py")

code("""
import requests
server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(30):
    try:
        print(requests.get("http://127.0.0.1:8000/health", timeout=1).json()); break
    except Exception:
        time.sleep(1)
""")

code("""
# Live call with the 6 real lines from the invoice
lines = [f"{i['description_en']} / {i['description_ar']}" for s in inv["sections"] for i in s["items"]]
req = {"descriptions": lines}
print("REQUEST:\\n", json.dumps(req, ensure_ascii=False, indent=2))
resp = requests.post("http://127.0.0.1:8000/predict", json=req).json()
print("\\nRESPONSE (status 200) - model", resp["model"], "from MLflow run", resp["run_id"])
display(pd.DataFrame(resp["predictions"])[["description", "category", "confidence", "needs_review"]])
print("\\nRaw JSON for one line:")
print(json.dumps(resp["predictions"][1], ensure_ascii=False, indent=2))
""")

code("""
# Same call with curl (what you would show in a terminal; Linux/Colab shell syntax)
if IN_COLAB:
    get_ipython().system("curl -s -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{\\"descriptions\\": [\\"Rebar 12 mm / حديد تسليح ١٢ مم\\"]}'")
""")

code("""
# Interactive Swagger docs: POST /predict -> Try it out -> Execute (a live call in the browser)
if IN_COLAB:
    from google.colab import output
    output.serve_kernel_port_as_window(8000, path="/docs", anchor_text="Open API docs (/docs)")
else:
    print("Open: http://127.0.0.1:8000/docs")
""")

md("""
Note the **waterproofing membrane**: the model must pick one of 5 classes it was trained on, but the
confidence is low, so `needs_review` catches it. That is the right behaviour for an out-of-taxonomy item.
""")

# ------------------------------------------------------------------ task 7
md("""
## 7 · Code & documentation
The README, labelling guide and prompts are written below. In Colab, the last cell downloads the whole project as a zip.
""")
writefile("README.md")
writefile("docs/LABELLING_GUIDE.md")
writefile("docs/PROMPTS.md")
writefile("requirements.txt")

code("""
# In Colab: download the whole project as a zip. Locally the project is already the git repo.
if IN_COLAB:
    import shutil
    from google.colab import files
    archive = shutil.make_archive(str(PROJECT.parent / "jubii7-track-a"), "zip", PROJECT.parent, PROJECT.name)
    files.download(archive)
else:
    print("Local run: project files are already in", PROJECT.name)
""")

code("""
# Stop the API and MLflow UI servers. In Colab they keep running for the live demo;
# run `server.terminate(); mlflow_ui.terminate()` yourself when you are done.
if not IN_COLAB:
    server.terminate(); mlflow_ui.terminate(); print("servers stopped")
else:
    print("Servers still running for the demo.")
""")
md("""
**Demo checklist:** (1) the MLflow run chart and table in §4 → open the MLflow UI → Compare the 3 runs;
(2) the live `/predict` call on the 6 invoice lines in §6 → open `/docs` → Try it out → Execute.
The response carries the MLflow `run_id`, so every prediction can be traced back to the run that made the model.
""")

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "colab": {"provenance": []}},
    "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "track_a_colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {out} ({len(cells)} cells)")
