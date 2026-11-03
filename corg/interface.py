"""Adaptive grounding interface (Section III-B of the paper).

For each model family, the input operator wraps a task prompt into the grounding format of the family, and the output
operator parses the response and maps the parsed coordinates to the pixels of the original image.
"""
import json
import math
import re

# Supported backbones. coord: "abs" = absolute pixels of the internally resized image, "norm1000" = values in [0, 1000].
MODELS = {
    "qwen2.5-vl": {"family": "qwen", "coord": "abs", "factor": 28},
    "qwen3-vl": {"family": "qwen", "coord": "norm1000", "factor": 32},
    "internvl3.5": {"family": "internvl", "coord": "norm1000", "factor": None},
    "mimo-vl": {"family": "mimo", "coord": "abs", "factor": 28},
}

# Six prompts for box hypotheses (P1 is the default prompt of each family) and the point prompt (Table III and Table S-I).
_MIMO_SCHEMA = 'Output a JSON in the format [{"bbox_2d": [...], "label": "{the_whole_description}"}, ...].'
PROMPTS = {
    "qwen": {
        "P1": "Locate {expr} in the image, output its bbox coordinates using JSON format.",
        "P2": "Locate {expr} in the image, output its bbox coordinates using JSON format. The bounding box must tightly "
              "enclose only the described object itself, excluding its shadow and any surrounding area.",
        "P3": "Output the precise bounding box of {expr}, following the object's own boundary only, using JSON format.",
        "P4": "Find {expr}. Give the smallest bounding box that contains the whole object and nothing else, in JSON format.",
        "P5": "Where is {expr} in this remote sensing image? Output its bbox coordinates using JSON format.",
        "P6": "Locate {expr} in the aerial image and output its bbox coordinates using JSON format. Do not include its shadow.",
        "Ppt": "Point to {expr} in the image, output its coordinates using JSON format.",
    },
    "internvl": {
        "P1": "Please provide the bounding box coordinate of the region this sentence describes: <ref>{expr}</ref>",
        "P2": "Please provide the bounding box coordinate of the region this sentence describes: <ref>{expr}</ref>. "
              "The box must tightly enclose only the object itself, excluding its shadow and surrounding area.",
        "P3": "Please provide the precise bounding box coordinate of the region this sentence describes, following the "
              "object's own boundary only: <ref>{expr}</ref>",
        "P4": "Please provide the smallest bounding box coordinate that contains the whole object this sentence describes "
              "and nothing else: <ref>{expr}</ref>",
        "P5": "In this remote sensing image, please provide the bounding box coordinate of the region this sentence "
              "describes: <ref>{expr}</ref>",
        "P6": "Please provide the bounding box coordinate of the region this sentence describes, excluding its shadow: "
              "<ref>{expr}</ref>",
        "Ppt": "Please provide the center point coordinate of the region this sentence describes: <ref>{expr}</ref>. "
               "Answer with [x, y] using coordinates normalized to 0-1000.",
    },
    "mimo": {
        "P1": 'Based on the description: "{expr}", locate all regions matching the description. ' + _MIMO_SCHEMA,
        "P2": 'Based on the description: "{expr}", locate all regions matching the description. ' + _MIMO_SCHEMA +
              " The bounding box must tightly enclose only the described object itself, excluding its shadow and any "
              "surrounding area.",
        "P3": "Output the precise bounding box of {expr}, following the object's own boundary only. " + _MIMO_SCHEMA,
        "P4": "Find {expr}. Give the smallest bounding box that contains the whole object and nothing else. " + _MIMO_SCHEMA,
        "P5": "Where is {expr} in this remote sensing image? " + _MIMO_SCHEMA,
        "P6": "Locate {expr} in the aerial image. Do not include its shadow. " + _MIMO_SCHEMA,
        "Ppt": "Point to {expr} in the image, output its coordinates using JSON format.",
    },
}
BOX_PROMPTS = ("P1", "P2", "P3", "P4", "P5", "P6")
MIMO_MIN_PIXELS, MIMO_MAX_PIXELS = 3136, 12845056


def family(model):
    return MODELS[model]["family"]


def prompt_text(model, prompt_id, expr):
    """Task prompt of a family for one expression. MiMo-VL requests end with /no_think, which disables reasoning."""
    text = PROMPTS[family(model)][prompt_id].format(expr=expr) if family(model) != "mimo" else \
        PROMPTS["mimo"][prompt_id].replace("{expr}", expr)
    if family(model) == "mimo":
        text = text.rstrip()
        text = text if text.endswith("/no_think") else text + " /no_think"
    return text


def chat_prompt(processor, model, text):
    """Input operator: wrap the task prompt into the chat format of the family (one image, image before text)."""
    if family(model) == "internvl":
        msgs = [{"role": "user", "content": "<image>\n" + text}]
    else:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def smart_resize(h, w, factor, min_pixels=4 * 28 * 28, max_pixels=16384 * 28 * 28):
    """Image size seen by Qwen-style processors: sides rounded to multiples of factor within a pixel range."""
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = math.floor(h / beta / factor) * factor
        wb = math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


# ----------------------------------------------------------------------------------------------- Qwen and InternVL
_NUM = r"-?\d+(?:\.\d+)?"


def _parse_box_text(text):
    """First box in the response: a bbox_2d field, then a bracketed list of four numbers, then a parenthesized one."""
    m = re.search(r'"bbox_2d"\s*:\s*\[\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')', text)
    if not m:
        m = re.search(r"\[\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\]", text)
    if not m:
        m = re.search(r"\(\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\)", text)
    return [float(v) for v in m.groups()] if m else None


def _parse_point_text(text):
    """A point [x, y]. A box returned to the point prompt (the regular case for InternVL3.5) is replaced by its center."""
    b = re.search(r"\[\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\]", text)
    if b and '"point_2d"' not in text:
        x1, y1, x2, y2 = (float(v) for v in b.groups())
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    m = re.search(r'"point_2d"\s*:\s*\[\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')', text) or \
        re.search(r"\[\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\]", text) or \
        re.search(r"\(\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\)", text)
    return [float(m.group(1)), float(m.group(2))] if m else None


def _scale(model, W, H):
    cfg = MODELS[model]
    if cfg["coord"] == "abs":
        rh, rw = smart_resize(H, W, cfg["factor"])
        return W / rw, H / rh
    return W / 1000.0, H / 1000.0


# --------------------------------------------------------------------------------------------------------- MiMo-VL
def _mimo_final(raw):
    """Text after the reasoning block; an unclosed block yields no answer."""
    if "</think>" in raw:
        final = raw.rsplit("</think>", 1)[1].strip()
        return "" if "<think>" in final else final
    return "" if "<think>" in raw else raw.strip()


def _json_roots(text):
    decoder, roots, i = json.JSONDecoder(), [], 0
    while i < len(text):
        if text[i] in "[{":
            try:
                value, used = decoder.raw_decode(text[i:])
                roots.append(value)
                i += used
                continue
            except ValueError:
                pass
        i += 1
    return roots


def _candidates(roots, keys, length):
    keyed, plain = [], []

    def walk(v):
        if isinstance(v, dict):
            for k, c in v.items():
                if k in keys:
                    keyed.append(c)
                elif isinstance(c, (dict, list)):
                    walk(c)
        elif isinstance(v, list):
            if v and all(not isinstance(x, (dict, list)) for x in v):
                plain.append(v)
            else:
                for c in v:
                    walk(c)
    for r in roots:
        walk(r)
    return keyed if keyed else [v for v in plain if len(v) == length] or plain


def _finite_vector(v, n):
    return (isinstance(v, (list, tuple)) and len(v) == n
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v))


def _mimo_parse(raw, kind):
    final = _mimo_final(raw)
    if not final:
        return None
    roots = _json_roots(final)
    keys = ("bbox_2d", "bbox", "box") if kind == "box" else ("point", "point_2d", "coordinates")
    cands = _candidates(roots, keys, 4 if kind == "box" else 2)
    if kind == "point" and not cands:
        cands = [[x["x"], x["y"]] for x in roots if isinstance(x, dict) and "x" in x and "y" in x]
    if kind == "point" and not any(_finite_vector(v, 2) for v in cands):
        boxes = [b for b in _candidates(roots, ("bbox_2d", "bbox", "box"), 4) if _finite_vector(b, 4) and b[2] > b[0] and b[3] > b[1]]
        if boxes:
            cands = [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes]
    for c in cands:
        if _finite_vector(c, 4 if kind == "box" else 2) and (kind != "box" or (c[2] > c[0] and c[3] > c[1])):
            return [float(x) for x in c]
    return None


def _mimo_size(W, H):
    rh, rw = smart_resize(H, W, 28, MIMO_MIN_PIXELS, MIMO_MAX_PIXELS)
    return rw, rh


# ---------------------------------------------------------------------------------------------- output operator
def parse_box(model, text, W, H):
    """Output operator for a box: parse the response and map it to the pixels of the W x H image, or None."""
    if family(model) == "mimo":
        raw = _mimo_parse(text, "box")
        if raw is None:
            return None
        rw, rh = _mimo_size(W, H)
        return [raw[0] * W / rw, raw[1] * H / rh, raw[2] * W / rw, raw[3] * H / rh]
    raw = _parse_box_text(text)
    if raw is None:
        return None
    sx, sy = _scale(model, W, H)
    x1, x2 = sorted((raw[0] * sx, raw[2] * sx))
    y1, y2 = sorted((raw[1] * sy, raw[3] * sy))
    clip = lambda v, hi: min(max(v, 0.0), float(hi))
    return [clip(x1, W), clip(y1, H), clip(x2, W), clip(y2, H)]


def parse_point(model, text, W, H):
    """Output operator for the target point, mapped to the pixels of the W x H image, or None."""
    if family(model) == "mimo":
        raw = _mimo_parse(text, "point")
        if raw is None:
            return None
        rw, rh = _mimo_size(W, H)
        return [raw[0] * W / rw, raw[1] * H / rh]
    p = _parse_point_text(text)
    if p is None:
        return None
    sx, sy = _scale(model, W, H)
    return [min(max(p[0] * sx, 0), W - 1), min(max(p[1] * sy, 0), H - 1)]
