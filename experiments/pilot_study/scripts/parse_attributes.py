from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import defaultdict

from common import RESULTS, load_config, read_csv, write_csv

SPECS = {
    "color": {
        "values": {
            "black": ["black"], "white": ["white", "ivory", "cream"], "gray": ["gray", "grey", "charcoal"],
            "red": ["red", "burgundy", "maroon", "wine"], "blue": ["blue", "navy", "teal", "turquoise", "aqua"],
            "green": ["green", "olive", "mint"], "yellow": ["yellow", "mustard", "gold"], "pink": ["pink", "blush", "rose"],
            "purple": ["purple", "violet", "lavender"], "brown": ["brown", "tan", "camel", "khaki"], "orange": ["orange", "coral"],
            "beige": ["beige", "nude"],
        },
        "triggers": ["color", "colour", "colored", "coloured", "shade", "black", "white", "ivory", "cream", "gray", "grey", "charcoal", "red", "burgundy", "maroon", "wine", "blue", "navy", "teal", "turquoise", "aqua", "green", "olive", "mint", "yellow", "mustard", "gold", "pink", "blush", "rose", "purple", "violet", "lavender", "brown", "tan", "camel", "khaki", "orange", "coral", "beige", "nude"],
    },
    "sleeve_length": {
        "values": {"sleeveless": ["sleeveless", "without sleeves"], "short": ["short sleeve", "short sleeves", "cap sleeve", "cap sleeves"], "three_quarter": ["three quarter sleeve", "three-quarter sleeve", "3/4 sleeve", "elbow length sleeve"], "long": ["long sleeve", "long sleeves"]},
        "triggers": ["sleeve", "sleeves", "sleeveless", "arm length"],
    },
    "garment_type": {
        "values": {"shirt": ["shirt", "button down", "button-down"], "t_shirt": ["t-shirt", "tee shirt", "tee"], "blouse": ["blouse"], "tank_top": ["tank top", "camisole"], "polo": ["polo"], "sweater": ["sweater", "cardigan", "pullover"], "tunic": ["tunic"], "dress": ["dress"]},
        "triggers": ["shirt", "top", "blouse", "t-shirt", "tee", "tank", "polo", "sweater", "cardigan", "pullover", "tunic", "dress", "garment", "clothing", "style"],
    },
    "pattern": {
        "values": {"solid": ["solid", "plain", "no pattern"], "striped": ["striped", "stripes", "stripe"], "floral": ["floral", "flower print", "flowers"], "plaid": ["plaid", "checkered", "checked", "gingham"], "graphic": ["graphic", "printed", "print", "logo"], "polka_dot": ["polka dot", "polka dots", "dot print"], "abstract": ["abstract pattern", "abstract print"]},
        "triggers": ["pattern", "print", "printed", "solid", "plain", "stripe", "striped", "floral", "plaid", "checkered", "checked", "gingham", "graphic", "logo", "polka dot", "dots", "abstract"],
    },
}

PRESERVE = re.compile(r"\b(same|keep|keeps|keeping|retain|retains|remain|remains|stays?|still|unchanged|without changing)\b", re.I)
EXCLUDE = re.compile(r"\b(without|avoid|avoiding|remove|removing|no longer|no|not|never|lack(?:ing)?|free of)\b", re.I)
MODIFY = re.compile(r"\b(change|make|turn|switch|replace|add|remove|swap|instead|rather than|more|less|shorter|longer|darker|lighter)\b", re.I)
COMPARATIVE = re.compile(r"\b(more|less|shorter|longer|darker|lighter|brighter|paler|bigger|smaller|same as|similar to)\b", re.I)
IMPLICIT_COMPARATIVE = {
    "color": ["lighter", "darker", "brighter", "paler", "more vivid", "less saturated"],
    "sleeve_length": ["shorter", "longer"],
    "garment_type": ["more formal", "more casual"],
    "pattern": ["more subtle", "less busy"],
}


def matches(text: str, terms: list[str]) -> list[str]:
    found = []
    for term in sorted(terms, key=len, reverse=True):
        pat = r"(?<![a-z0-9])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        if re.search(pat, text, re.I):
            found.append(term)
    return found


def parse_one(attribute: str, text: str) -> dict[str, str]:
    spec = SPECS[attribute]
    norm = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    domain_mentioned = any(re.search(r"(?<![a-z0-9])" + re.escape(t).replace(r"\ ", r"\s+") + r"(?![a-z0-9])", norm) for t in spec["triggers"])
    all_terms = {term: value for value, terms in spec["values"].items() for term in terms}
    found_terms = matches(norm, list(all_terms))
    found_values = []
    for term in found_terms:
        value = all_terms[term]
        if value not in found_values:
            found_values.append(value)
    out = {"attribute": attribute, "operation": "unspecified", "requested_value": "", "exclude_values": "", "parse_status": "unspecified", "parse_reason": "attribute not mentioned"}
    implicit_comparative = any(re.search(r"(?<![a-z0-9])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![a-z0-9])", norm) for term in IMPLICIT_COMPARATIVE[attribute])
    if not domain_mentioned and not found_values and not implicit_comparative:
        return out
    if COMPARATIVE.search(norm):
        out.update(operation="unknown", parse_status="unknown", parse_reason="comparative or relative wording requires a reliable reference-based measurement")
        return out
    if not found_values and domain_mentioned:
        if PRESERVE.search(norm):
            out.update(operation="preserve", parse_status="parsed", parse_reason="explicit preservation cue; evaluate relative to reference")
        elif EXCLUDE.search(norm):
            out.update(operation="unknown", parse_status="unknown", parse_reason="exclusion stated but no discrete excluded value can be identified")
        else:
            out.update(operation="unknown", parse_status="unknown", parse_reason="attribute mentioned without a discrete value or clear state operation")
        return out
    excluded_terms = []
    transition = re.search(r"\b(instead of|rather than)\b", norm)
    for term in found_terms:
        before = norm[:norm.find(term)]
        suffix = before[-28:]
        if (term == "without sleeves"):
            continue
        after_transition = transition and norm.find(term) >= transition.end()
        if (after_transition or (EXCLUDE.search(suffix) and not re.search(r"\b(not only|not just)\b", suffix))):
            value = all_terms[term]
            if value not in excluded_terms:
                excluded_terms.append(value)
    preserve = bool(PRESERVE.search(norm))
    explicit_modify = bool(MODIFY.search(norm)) or bool(found_values and re.search(r"\b(but|instead of|to a|into a|with a|with an)\b", norm))
    desired = [v for v in found_values if v not in excluded_terms]
    if preserve and not explicit_modify:
        out.update(operation="preserve", requested_value="|".join(desired), exclude_values="|".join(excluded_terms), parse_status="parsed", parse_reason="explicit preservation cue")
    elif excluded_terms and not desired:
        out.update(operation="exclude", requested_value="", exclude_values="|".join(excluded_terms), parse_status="parsed", parse_reason="explicit negative/exclusion cue")
    elif desired and (explicit_modify or not excluded_terms):
        if len(set(desired)) > 1:
            out.update(operation="unknown", parse_status="unknown", parse_reason="multiple possible requested values for one attribute")
        else:
            out.update(operation="modify", requested_value=desired[0], exclude_values="|".join(excluded_terms), parse_status="parsed", parse_reason="discrete value stated in the modification text")
    else:
        out.update(operation="unknown", parse_status="unknown", parse_reason="mixed or ambiguous positive and negative attribute wording")
    return out


def parse_query(row: dict[str, str]) -> list[dict[str, str]]:
    text = f"{row.get('caption_1', '')} {row.get('caption_2', '')}".strip()
    parsed = []
    for attr in SPECS:
        result = parse_one(attr, text)
        parsed.append({"query_id": row["query_id"], "split": row.get("split", ""), "caption_1": row.get("caption_1", ""), "caption_2": row.get("caption_2", ""), **result,
                       "attribute_evaluable": result["parse_status"] == "parsed" and result["operation"] in {"modify", "preserve", "exclude"}})
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservative English attribute operation parser")
    parser.add_argument("--input", default=str(RESULTS / "query_inputs_sampled.csv"))
    parser.add_argument("--output", default="attribute_constraints.csv")
    args = parser.parse_args()
    rows = read_csv(args.input)
    parsed = [r for query in rows for r in parse_query(query)]
    out = RESULTS / args.output if not Path(args.output).is_absolute() else Path(args.output)
    write_csv(out, parsed)
    counts = defaultdict(int)
    for row in parsed:
        counts[f"operation:{row['operation']}"] += 1
        counts[f"status:{row['parse_status']}"] += 1
        counts[f"attribute_evaluable:{row['attribute_evaluable']}"] += 1
    print(dict(counts))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
