"""Task 1 - Data understanding: profile the extracted invoice, recalculate
every total and list every problem found.

Usage:  python -m invoice_ml.check_invoice [data/invoice_extracted.json]
Writes: reports/invoice_profile.csv, reports/invoice_problems.csv,
        reports/invoice_clean.json
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from invoice_ml.text_utils import (arabic_words_to_number, digit_script, parse_number,
                                   to_western_digits)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TOL = 0.01  # AED rounding tolerance
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


# ---------------------------------------------------------------- profiling
def _infer_type(key: str, value) -> str:
    if value is None or value == "":
        return "missing"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    s = to_western_digits(str(value))
    if "date" in key:
        return "date"
    if key in {"trn", "invoice_no", "po_no", "phone", "vehicle"}:
        return "id"
    if re.fullmatch(r"\(?[\d.,\s%]+\)?", s):
        return "number"
    return "text"


def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not k.startswith("_"):
                yield from _walk(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def profile(inv: dict) -> pd.DataFrame:
    rows = []
    for path, value in _walk(inv):
        key = re.sub(r"\[\d+\]", "", path).split(".")[-1]
        rows.append({"field": path, "value": value, "type": _infer_type(key, value),
                     "digits": digit_script(value if isinstance(value, str) else None)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- helpers
def parse_date(s: str | None) -> date | None:
    if not s:
        return None
    s = to_western_digits(s).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def money(x: float) -> float:
    return round(x + 1e-9, 2)


def days_in_terms(text: str | None) -> int | None:
    m = re.search(r"(\d+)\s*(?:days|يوم)", to_western_digits(text or ""))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- checks
def check(inv: dict) -> tuple[dict, pd.DataFrame]:
    problems: list[dict] = []

    def flag(severity, field, issue, printed=None, expected=None):
        problems.append({"severity": severity, "field": field, "issue": issue,
                         "printed": printed, "expected": expected})

    doc, sup, cus = inv["document"], inv["supplier"], inv["customer"]
    inv_date = parse_date(doc.get("invoice_date"))

    # --- header
    for party, data in (("supplier", sup), ("customer", cus)):
        trn = data.get("trn")
        if not trn:
            flag("high", f"{party}.trn", "TRN missing")
        elif len(re.sub(r"\D", "", to_western_digits(trn))) != 15:
            flag("high", f"{party}.trn", "UAE TRN must be 15 digits", printed=trn)
    ticked = doc.get("payment_methods_ticked") or []
    if len(ticked) != 1:
        flag("medium", "document.payment", f"{len(ticked)} payment methods ticked "
             f"({', '.join(ticked) or 'none'}) - Cash and Credit contradict each other",
             printed=ticked)
    for k in ("po_no",):
        if digit_script(doc.get(k)) in ("arabic", "mixed"):
            flag("info", f"document.{k}", "Arabic-Indic digits in an ID - normalise before "
                 "matching to the PO system", printed=doc[k],
                 expected=to_western_digits(doc[k]))

    # --- line items
    sections_out, total_correct, total_printed_lines = [], 0.0, 0.0
    for sec in inv["sections"]:
        s_correct = s_printed = 0.0
        items_out = []
        for it in sec["items"]:
            ref = f"line {it['line']} ({it['description_en']})"
            q_ord, q_del, price, amount = (parse_number(it.get(k)) for k in
                                           ("qty_ordered", "qty_delivered", "unit_price", "amount"))
            for k in ("qty_ordered", "qty_delivered", "unit_price", "amount"):
                if digit_script(it.get(k)) == "arabic":
                    flag("info", f"{ref}.{k}", "Arabic-Indic digits, normalised",
                         printed=it[k], expected=parse_number(it[k]))
            billed_qty = q_del if q_del is not None else q_ord
            if q_ord is not None and q_del is not None and q_ord != q_del:
                flag("medium", f"{ref}.qty", f"Ordered {q_ord:g} but delivered {q_del:g}: "
                     "billed on delivered qty, balance must not be billed twice later",
                     printed=f"{q_ord:g} / {q_del:g}")
            if billed_qty is None or price is None:
                flag("high", ref, "Cannot recalculate: qty or unit price missing")
                correct = amount or 0.0
            else:
                correct = money(billed_qty * price)
                if amount is None:
                    flag("high", f"{ref}.amount", "Amount missing", expected=correct)
                elif abs(correct - amount) > TOL:
                    hint = ""
                    if sorted(f"{amount:.2f}") == sorted(f"{correct:.2f}"):
                        hint = " - same digits, looks like a transposition"
                    flag("high", f"{ref}.amount",
                         f"qty x price = {billed_qty:g} x {price:,.2f}{hint}",
                         printed=amount, expected=correct)
            s_correct += correct
            s_printed += amount or 0.0
            items_out.append({**it, "qty_ordered": q_ord, "qty_delivered": q_del,
                              "unit_price": price, "amount_printed": amount,
                              "amount": correct})

        sub_p = parse_number(sec.get("subtotal_printed"))
        if sub_p is None or abs(sub_p - money(s_correct)) > TOL:
            carried = sub_p is not None and abs(sub_p - money(s_printed)) <= TOL
            flag("high", f"{sec['name_en']}.subtotal", "Section subtotal wrong" +
                 (" (adds up the printed lines, so the line error carried through)" if carried
                  else " (does not match the printed lines either)"),
                 printed=sub_p, expected=money(s_correct))
        total_correct += s_correct
        total_printed_lines += s_printed
        sections_out.append({"name_en": sec["name_en"], "name_ar": sec["name_ar"],
                             "items": items_out, "subtotal": money(s_correct),
                             "subtotal_printed": sub_p})

    # --- totals: check each step against the PRINTED previous step (where did it
    #     break?) and against the full recalculation (what should it be?)
    t = inv["totals_printed"]
    p = {k: parse_number(t.get(k)) for k in
         ("total_before_discount", "discount", "taxable_amount", "vat_amount", "grand_total")}
    p["discount"] = abs(p["discount"] or 0.0)        # printed as "(155.20)"
    vat_rate = (parse_number(t.get("vat_rate")) or 5.0) / 100
    if abs(vat_rate - 0.05) > 1e-9:
        flag("high", "totals.vat_rate", "UAE standard VAT rate is 5%", printed=t["vat_rate"])

    m = re.search(r"([\d.]+)\s*%\s*on\s*(.+)", t.get("discount_label") or "", re.I)
    disc_rate = float(m.group(1)) / 100 if m else 0.0
    disc_target = m.group(2).strip().lower() if m else ""
    disc_secs = [s for s in sections_out if disc_target and disc_target in s["name_en"].lower()]
    disc_base_correct = sum(s["subtotal"] for s in disc_secs) if disc_secs else total_correct
    disc_base_printed = (sum(s["subtotal_printed"] for s in disc_secs) if disc_secs
                         else p["total_before_discount"])
    if disc_secs and any("delivery" in i["description_en"].lower()
                         for s in disc_secs for i in s["items"]):
        flag("low", "totals.discount", "Discount base includes the delivery charge - "
             "confirm the discount is meant to apply to services too")

    c = {}
    c["total_before_discount"] = money(total_correct)
    c["discount"] = money(disc_base_correct * disc_rate)
    c["taxable_amount"] = money(c["total_before_discount"] - c["discount"])
    c["vat_amount"] = money(c["taxable_amount"] * vat_rate)
    c["grand_total"] = money(c["taxable_amount"] + c["vat_amount"])

    printed_chain = {
        "total_before_discount": sum(s["subtotal_printed"] or 0 for s in sections_out),
        "discount": money(disc_base_printed * disc_rate),
        "taxable_amount": money(p["total_before_discount"] - p["discount"]),
        "vat_amount": money(p["taxable_amount"] * vat_rate),
        "grand_total": money(p["taxable_amount"] + p["vat_amount"]),
    }
    for k in c:
        if abs(p[k] - printed_chain[k]) > TOL:
            flag("high", f"totals.{k}", "Arithmetic error on the page: does not follow from "
                 "the printed figures above it", printed=p[k], expected=printed_chain[k])
        elif abs(p[k] - c[k]) > TOL:
            flag("high", f"totals.{k}", "Wrong after recalculation (error carried from above)",
                 printed=p[k], expected=c[k])

    words = inv.get("amount_in_words_ar")
    if words:
        w = arabic_words_to_number(words)
        if w is None:
            flag("medium", "amount_in_words", "Could not parse amount in words", printed=words)
        elif abs(w - p["grand_total"]) > TOL:
            hint = (" - same digits, looks like a transposition"
                    if sorted(f"{w:.2f}") == sorted(f"{p['grand_total']:.2f}") else "")
            flag("high", "amount_in_words", f"Words say {w:,.2f} but the figure says "
                 f"{p['grand_total']:,.2f}{hint}", printed=w, expected=c["grand_total"])

    # --- deliveries vs invoice
    for d in inv.get("deliveries", []):
        ref = f"delivery {d['delivery']}"
        dd = parse_date(d.get("planned_date"))
        if d.get("planned_date") and dd is None:
            flag("high", f"{ref}.planned_date", "Invalid calendar date", printed=d["planned_date"])
        elif dd and inv_date and dd > inv_date:
            status = "but is marked received" if d.get("received_by") else "and is not received"
            flag("medium", f"{ref}.planned_date", f"Delivery is after the invoice date {status}; "
                 "its items are billed as already delivered", printed=d["planned_date"])
        if not d.get("received_by"):
            flag("medium", f"{ref}.received_by", "No receiver - goods not confirmed delivered"
                 + (f" ({d['note']})" if d.get("note") else ""))
    trips = len(inv.get("deliveries", []))
    del_lines = [i for s in sections_out for i in s["items"]
                 if "delivery" in i["description_en"].lower()]
    for i in del_lines:
        if i["qty_delivered"] and trips and i["qty_delivered"] != trips:
            flag("low", f"line {i['line']} (Delivery charge)", f"Charged {i['qty_delivered']:g} "
                 f"trip but the delivery table shows {trips} trips", printed=i["qty_delivered"],
                 expected=trips)
    if "sharjah only" in (inv.get("terms_en") or "").lower() and \
            "عجمان" in (doc.get("delivery_to") or ""):
        flag("low", "document.delivery_to", "Delivery address is in Ajman but prices only "
             "include delivery inside Sharjah - check the delivery charge")

    # --- terms, signatures, stamp
    d_en, d_ar = days_in_terms(inv.get("terms_en")), days_in_terms(inv.get("terms_ar"))
    if d_en and d_ar and d_en != d_ar:
        flag("high", "terms", f"English terms say {d_en} days, Arabic terms say {d_ar} days",
             printed=f"{d_en} / {d_ar}")
    if inv_date and d_en:
        flag("info", "terms.due_date", "Due date implied by terms",
             expected=f"{inv_date + timedelta(days=d_en)} (EN) / "
                      f"{inv_date + timedelta(days=d_ar or d_en)} (AR)")
    for k, v in (inv.get("signatures") or {}).items():
        if not v:
            flag("medium", f"signatures.{k}", "Blank")
    stamp = inv.get("stamp") or {}
    if stamp.get("present") and not (inv.get("signatures") or {}).get("customer_signature"):
        flag("low", "stamp", "Supplier stamp present but customer signature blank")

    clean = {"document": {**doc, "invoice_date": str(inv_date)},
             "supplier": sup, "customer": cus, "sections": sections_out,
             "totals_printed": p, "totals_recalculated": c,
             "deliveries": inv.get("deliveries")}
    df = pd.DataFrame(problems)
    df = df.sort_values("severity", key=lambda s: s.map(SEVERITY_ORDER), kind="stable")
    return clean, df.reset_index(drop=True)


def main(path: str | None = None) -> None:
    src = Path(path) if path else ROOT / "data" / "invoice_extracted.json"
    inv = json.loads(src.read_text(encoding="utf-8"))
    REPORTS.mkdir(exist_ok=True)

    prof = profile(inv)
    prof.to_csv(REPORTS / "invoice_profile.csv", index=False, encoding="utf-8-sig")
    clean, problems = check(inv)
    (REPORTS / "invoice_clean.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    problems.to_csv(REPORTS / "invoice_problems.csv", index=False, encoding="utf-8-sig")

    print(f"Profiled {len(prof)} fields: {prof['type'].value_counts().to_dict()}")
    print(f"Digit scripts: {prof['digits'].value_counts().to_dict()}")
    print("\nTotals (printed -> recalculated):")
    for k, v in clean["totals_recalculated"].items():
        print(f"  {k:22} {clean['totals_printed'][k]:>12,.2f} -> {v:>12,.2f}")
    print(f"\n{len(problems)} problems ({problems['severity'].value_counts().to_dict()}):")
    for r in problems.itertuples():
        extra = ""
        if r.printed is not None or r.expected is not None:
            extra = f"  [printed={r.printed} expected={r.expected}]"
        print(f"  {r.severity.upper():6} {r.field}: {r.issue}{extra}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
