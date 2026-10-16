"""Indonesian date-string cleaning shared by the notebook (load_news) and diagnose scripts.

Single copy on purpose: the portal date formats drift across re-scrapes, and
the trainer + the corpus audit must parse them identically.
"""
import re

_ID_MONTH = {"januari": "01", "februari": "02", "maret": "03", "april": "04", "mei": "05",
    "juni": "06", "juli": "07", "agustus": "08", "september": "09", "oktober": "10",
    "november": "11", "desember": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "jun": "06", "jul": "07",
    "agu": "08", "ags": "08", "sep": "09", "okt": "10", "nov": "11", "des": "12"}


def clean_day(s):
    s = str(s).lower().replace(" wib", "").replace("|", " ").replace("/", " ")
    s = re.sub(r"\s*[+-]\d{2}:?\d{2}\s*$", "", s)  # drop tz offset (RFC +0700 or ISO +07:00), keep Jakarta wall time
    s = re.sub(r"(?<=\d)t(?=\d)", " ", s)  # ISO T separator -> space (string is already lowercased above)
    s = re.sub(r"^(senin|selasa|rabu|kamis|jumat|sabtu|minggu),?\s*", "", s).strip()
    for _k, _v in _ID_MONTH.items():
        s = re.sub(rf"\b{_k}\b", _v, s)
    return s
