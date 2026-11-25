# Prompts used

## 1. Extraction (scan -> JSON), run on a vision LLM
```
You are transcribing a scanned bilingual (Arabic/English) UAE tax invoice + delivery note.
Return ONLY JSON matching this schema: <paste the keys of data/invoice_extracted.json>.
Rules:
- Copy every value EXACTLY as printed. Do NOT fix arithmetic, do NOT convert Arabic digits,
  do NOT fill gaps. Missing or illegible -> null, and list it in "_illegible".
- Keep line items inside their printed groups, with the printed group subtotal.
- Transcribe the handwritten note and describe the stamp (text, what it covers).
- For each field you are unsure of, add it to "_low_confidence" with a reason.
```
Why: the checker must see the *printed* errors. An LLM that "helpfully" corrects totals hides exactly the mistakes the finance team needs to catch.

## 2. Synthetic labelled lines
```
Generate 60 invoice line descriptions from a UAE building-materials supplier,
12 per category: cement, steel, aggregates, blocks, services. Half English, half Arabic
(Gulf usage), with realistic sizes/units (kg, mm, m3, truck, trip). Include hard cases:
"concrete block", "rebar cutting service", "concrete pump hire", "sand per truck".
Output CSV: text,lang,ai_label
```
Then every line was reviewed by hand against `LABELLING_GUIDE.md`; the AI label is kept in
`ai_label`, the corrected one in `label`, and every change explained in `review_note`.
Typical AI error: labelling by a keyword ("concrete", "truck", "خرسانة") instead of the item.

## 3. Coding assistant
Claude Code was used to scaffold the repo (checker, training, MLflow logging, FastAPI).
All logic, thresholds and outputs were read and run before submitting.
