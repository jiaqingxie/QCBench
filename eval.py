import json
import os
import re
from decimal import Decimal, getcontext

getcontext().prec = 50

def _clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)

    s = re.sub(r'[\x00-\x1F\x7F]', '', s)

    s = s.replace('$', '')
    s = s.replace(r'\left', '').replace(r'\right', '')
    s = s.replace(r'\,', '').replace(' ', '')
    s = s.replace(',', '')

    s = s.replace('−', '-')
    s = s.replace('–', '-').replace('—', '-')
    s = s.replace('×', 'x')
    s = s.replace(r'\times', 'x')
    s = s.replace(r'\cdot', 'x')
    s = s.replace('∙', 'x').replace('·', 'x')

    s = re.sub(r'(?<!\\)frac', r'\\frac', s)

    s = re.sub(r'\\(?:mathrm|operatorname|text|textrm|mathbf|mathit|mathrmbf)\{([^{}]*)\}', r'\1', s)

    s = re.sub(r'\\boxed\{([^{}]*)\}', r'\1', s)

    return s

def _to_decimal(s: str):
    s = _clean_text(s)
    if not s:
        return None

    if re.fullmatch(r'[+-]?\d+(\.\d+)?', s):
        try:
            return Decimal(s)
        except Exception:
            pass

    if re.fullmatch(r'[+-]?\d+(\.\d+)?[eE][+-]?\d+', s):
        try:
            return Decimal(s)
        except Exception:
            pass

    m = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)x10\^?\{?([+-]?\d+)\}?', s)
    if m:
        coef = Decimal(m.group(1))
        exp = int(m.group(2))
        return coef * (Decimal(10) ** exp)

    m = re.fullmatch(r'10\^?\{?([+-]?\d+)\}?', s)
    if m:
        exp = int(m.group(1))
        return Decimal(10) ** exp

    m = re.fullmatch(r'\\frac\{([+-]?\d+(?:\.\d+)?)\}\{([+-]?\d+(?:\.\d+)?)\}', s)
    if m:
        num = Decimal(m.group(1))
        den = Decimal(m.group(2))
        if den == 0:
            return None
        return num / den

    m = re.fullmatch(r'([+-])?(\d+)\\frac\{(\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}', s)
    if m:
        sign = -1 if m.group(1) == '-' else 1
        k = Decimal(m.group(2))
        num = Decimal(m.group(3))
        den = Decimal(m.group(4))
        if den == 0:
            return None
        return Decimal(sign) * (k + num / den)

    m = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)', s)
    if m:
        num = Decimal(m.group(1))
        den = Decimal(m.group(2))
        if den == 0:
            return None
        return num / den

    return None

def is_numeric_equal(gt, answer, rel_tol=1e-6, zero_tol=1e-12) -> int:
    a = _to_decimal(gt)
    b = _to_decimal(answer)
    if a is None or b is None:
        return 0

    diff = abs(a - b)
    max_ab = max(abs(a), abs(b))

    if max_ab == 0:
        return 1

    if abs(a) == 0 or abs(b) == 0:
        return 1 if diff <= Decimal(str(zero_tol)) else 0

    rel_ok = diff <= (Decimal(str(rel_tol)) * max_ab)
    return 1 if rel_ok else 0

def is_numeric_equal_pro(gt, answer) -> int:
    a = _to_decimal(gt)
    b = _to_decimal(answer)

    if a is None or b is None:
        return 0

    gt_str = str(gt).strip()
    match = re.search(r'\.(\d+)', gt_str)
    decimals = len(match.group(1)) if match else 0

    b_rounded = b.quantize(Decimal('1e-{0}'.format(decimals)))

    return 1 if a == b_rounded else 0

def extract_boxed(text: str):
    match = re.search(r'\\boxed\{([^{}]+)\}', text)
    if match:
        return match.group(1).strip()
    return ""

if __name__ == "__main__":
    input_path = "your_result_path"
    output_path = "your_result_path"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scored_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            pred = extract_boxed(item.get("llm_answer", ""))
            gt = item.get("gt_answer", "")
            score = is_numeric_equal_pro(gt, pred)
            item["pred_answer"] = pred
            item["score"] = score
            
            original_class = item.get("class", "")
            if original_class in ["Biochemistry", "Organic"]:
                item["class"] = "BOC"
            
            scored_data.append(item)
    scored_data.sort(key=lambda x: x["index"])
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scored_data, f, indent=2, ensure_ascii=False) 