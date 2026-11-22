"""Token overlap diagnostics computed alongside semantic answer correctness."""

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence


def tokens(value: str) -> list[str]:
    text = unicodedata.normalize("NFKC", value).casefold()
    return re.findall("\\w+(?:[.\\-]\\w+)*|[<>≤≥=%]", text, flags=re.UNICODE)


def answer_scores(prediction: str, answers: Sequence[str]) -> dict[str, float]:
    predicted = tokens(prediction)
    f1 = 0.0
    if not predicted:
        return {"f1": 0.0}
    p = Counter(predicted)
    for answer in answers:
        gold = tokens(answer)
        overlap = sum((p & Counter(gold)).values())
        f1 = max(f1, 2 * overlap / (len(predicted) + len(gold)))
    return {"f1": f1}
