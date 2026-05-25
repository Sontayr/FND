from __future__ import annotations

import os
import re
import time
import math
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# CONFIG
# =========================
# Включатели модулей (можешь включать по одному)
ENABLE_FACTCHECK = True
ENABLE_RSS = True  # <-- поставь True, когда захочешь использовать RSS

FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY", "")

RSS_FEEDS = [
    ("Lenta", "https://lenta.ru/rss/news"),
    ("Kommersant", "https://www.kommersant.ru/RSS/news.xml"),
    # Можно добавить позже:
    # ("RBC", "https://rssexport.rbc.ru/rbcnews/news/30/full.rss"),
]


MODEL_MAX_TOKENS = 256
CHUNK_STRIDE_TOKENS = 64


W_RUBERT = 0.40
W_CLICKBAIT = 0.15
W_FACTCHECK = 0.30
W_NEWS = 0.15


THR_TRUE = 0.65
THR_FALSE = 0.35

# =========================
# PATHS (под твою структуру)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "ml", "models", "rubert-base-fakenews"))

app = FastAPI(title="FakeNews Detector API", version="1.1")


# =========================
# API Models
# =========================
class PredictRequest(BaseModel):
    text: str = Field(..., description="Текст новости/сообщения для проверки (любой длины)")


class EvidenceItem(BaseModel):
    source: str
    title: Optional[str] = None
    url: Optional[str] = None
    rating: Optional[str] = None
    score: Optional[float] = None
    note: Optional[str] = None


class SignalItem(BaseModel):
    name: str
    value: float      # 0..1 (где 1 = усиливает "правда")
    weight: float
    detail: Optional[str] = None


class PredictResponse(BaseModel):
    # обратная совместимость со старым Android
    label: str
    score: float

    # расширенный ответ
    verdict: str
    truth_score: float
    signals: List[SignalItem] = []
    suspicious_fragments: List[str] = []
    evidence: List[EvidenceItem] = []
    meta: Dict[str, Any] = {}


# =========================
# Load model once
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# =========================
# Helpers: text splitting
# =========================
_sentence_split_re = re.compile(r'(?<=[.!?…])\s+')

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = _sentence_split_re.split(text)
    sents = [s.strip() for s in sents if len(s.strip()) >= 5]
    return sents


def tokenize_to_ids(text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def chunk_token_ids(token_ids: List[int], max_len: int, stride: int) -> List[List[int]]:
    if len(token_ids) <= max_len:
        return [token_ids]
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_len, len(token_ids))
        chunks.append(token_ids[start:end])
        if end == len(token_ids):
            break
        start = end - stride
        if start < 0:
            start = 0
    return chunks


def ids_to_text(token_ids: List[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


# =========================
# Model inference
# =========================
@torch.no_grad()
def p_fake_for_text(text: str) -> float:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MODEL_MAX_TOKENS,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze(0)
    return float(probs[1].item())  # class 1 = fake


def style_score_long_text(text: str) -> Tuple[float, List[Tuple[str, float]]]:
    token_ids = tokenize_to_ids(text)
    chunks = chunk_token_ids(token_ids, MODEL_MAX_TOKENS, CHUNK_STRIDE_TOKENS)

    scored: List[Tuple[str, float]] = []
    for ch in chunks:
        ch_text = ids_to_text(ch)
        pf = p_fake_for_text(ch_text)
        scored.append((ch_text, pf))

    pf_max = max(p for _, p in scored) if scored else 0.5
    pf_mean = sum(p for _, p in scored) / len(scored) if scored else 0.5

    # чуть сглаживаем: важнее max, но учитываем и mean
    style_fake = float(0.7 * pf_max + 0.3 * pf_mean)

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:3]
    return style_fake, top


def suspicious_sentences(text: str, top_k: int = 3) -> List[str]:
    sents = split_sentences(text)
    if not sents:
        return []

    scored = []
    max_sents = 50  # чтобы не тормозить на огромных текстах
    for s in sents[:max_sents]:
        pf = p_fake_for_text(s)
        scored.append((s, pf))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:top_k]]


# =========================
# Clickbait / heuristic signal
# =========================
CLICKBAIT_PATTERNS = [
    r"\bсрочно\b",
    r"\bсенсаци\w*\b",
    r"\bшок\b",
    r"\bвсе\s+в\s+панике\b",
    r"\bвы\s+не\s+поверите\b",
    r"\bтолько\s+сейчас\b",
    r"\bскрыва(ют|ли)\b",
    r"\bправда\s+которую\s+не\s+говорят\b",
    r"\bнемедленно\b",
    r"\bсмотри\s+пока\s+не\s+удалили\b",
]
_clickbait_re = re.compile("|".join(CLICKBAIT_PATTERNS), re.IGNORECASE)

def clickbait_fake_score(text: str) -> Tuple[float, str]:
    """
    Возвращает (fake_score 0..1, detail)
    """
    t = text.strip()
    if not t:
        return 0.0, "пустой текст"

    # 1) ключевые слова
    hits = _clickbait_re.findall(t)
    hit_count = len(hits)

    # 2) капс
    letters = [ch for ch in t if ch.isalpha()]
    caps = [ch for ch in letters if ch.isupper()]
    caps_ratio = (len(caps) / len(letters)) if letters else 0.0

    # 3) восклицания/вопросы
    exc = t.count("!")
    qst = t.count("?")

    # 4) длина
    length = len(t)

    # скейлинг в 0..1 (простая формула)
    s = 0.0
    s += min(1.0, hit_count / 2.0) * 0.50
    s += min(1.0, caps_ratio / 0.35) * 0.25
    s += min(1.0, (exc + qst) / 6.0) * 0.20
    if length < 30:
        s += 0.05  # очень короткий кликбейтный заголовок

    s = max(0.0, min(1.0, s))

    detail = f"hits={hit_count}, caps_ratio={caps_ratio:.2f}, !+?={exc+qst}"
    return float(s), detail


# =========================
# FactCheck (optional) - placeholder
# =========================
def factcheck_search(claim: str) -> List[EvidenceItem]:
    if not ENABLE_FACTCHECK or not FACTCHECK_API_KEY:
        return []

    try:
        import requests
    except Exception:
        return []

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "key": FACTCHECK_API_KEY,
        "query": claim,
        "languageCode": "ru",
        "pageSize": 5,
        "maxAgeDays": 3650,
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    items: List[EvidenceItem] = []

    for claim_obj in data.get("claims", []):
        text = claim_obj.get("text", "")
        claim_reviews = claim_obj.get("claimReview", []) or []

        for review in claim_reviews:
            publisher = (((review.get("publisher") or {}).get("name")) or "FactCheck")
            title = review.get("title") or text or "Найден фактчек"
            review_url = review.get("url")
            textual_rating = review.get("textualRating")
            language_code = review.get("languageCode")

            note_parts = []
            if text:
                note_parts.append(f"claim: {text[:220]}")
            if language_code:
                note_parts.append(f"lang: {language_code}")

            score = map_factcheck_rating_to_score(
                textual_rating,
                title=title,
                note=" | ".join(note_parts) if note_parts else None
            )

            items.append(
                EvidenceItem(
                    source=publisher,
                    title=title,
                    url=review_url,
                    rating=textual_rating,
                    score=score,
                    note=" | ".join(note_parts) if note_parts else "найдено в фактчекинге"
                )
            )

    # убираем дубли по (source, title, url)
    uniq = []
    seen = set()
    for it in items:
        key = (it.source or "", it.title or "", it.url or "")
        if key not in seen:
            seen.add(key)
            uniq.append(it)

    return uniq[:8]


def map_factcheck_rating_to_score(
    rating: Optional[str],
    title: Optional[str] = None,
    note: Optional[str] = None
) -> Optional[float]:
    parts = [rating or "", title or "", note or ""]
    text = " | ".join(parts).strip().lower()

    if not text:
        return None

    false_patterns = [
        "false", "mostly false", "pants on fire", "misleading",
        "лож", "ложь", "неправ", "фейк", "опроверг", "манипуляц",
        "поддел", "фальш", "не соответствует действительности",
        "без доказательств", "спотворено", "неправда"
    ]

    true_patterns = [
        "true", "mostly true",
        "правд", "подтверж", "верно", "достоверно"
    ]

    mixed_patterns = [
        "partly", "partially", "mixed", "mixture", "half true",
        "частично", "сомнительно", "manipulation", "needs context"
    ]

    if any(p in text for p in false_patterns):
        return 0.15
    if any(p in text for p in true_patterns):
        return 0.85
    if any(p in text for p in mixed_patterns):
        return 0.50

    return 0.50


def extract_claims(text: str, max_claims: int = 6) -> List[str]:
    sents = split_sentences(text)
    if not sents:
        return []
    key_re = re.compile(r"\b(заявил|сообщил|произошл|подтверд|опроверг|сегодня|вчера|с\s+\d{4}|в\s+\d{4})\b", re.IGNORECASE)
    num_re = re.compile(r"\d")
    picks = []
    for s in sents:
        score = 0
        if key_re.search(s):
            score += 2
        if num_re.search(s):
            score += 1
        if 40 <= len(s) <= 220:
            score += 1
        if score >= 2:
            picks.append((s, score))
    if not picks:
        return sents[:max_claims]
    picks.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in picks[:max_claims]]


# =========================
# RSS verification (optional, real implementation)
# =========================
def normalize_tokens(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-zа-я0-9\s]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split(" ") if len(t) >= 4]
    return toks


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def fetch_rss_titles(url: str, limit: int = 25) -> List[Tuple[str, str]]:
    """
    Возвращает список (title, link). Требует requests.
    """
    try:
        import requests
        import xml.etree.ElementTree as ET
    except Exception:
        return []

    try:
        r = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        xml = r.text
        root = ET.fromstring(xml)

        items = []
        # RSS: обычно channel/item
        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el = item.find("link")
            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            link = link_el.text.strip() if link_el is not None and link_el.text else ""
            if title:
                items.append((title, link))
            if len(items) >= limit:
                break
        return items
    except Exception:
        return []


def rss_verify(text: str) -> List[EvidenceItem]:
    if not ENABLE_RSS:
        return []

    # Берём краткий "запрос": первые 1-2 предложения или первые 140 символов
    sents = split_sentences(text)
    query = sents[0] if sents else text[:140]
    q_toks = normalize_tokens(query)

    evidence: List[EvidenceItem] = []
    if not q_toks:
        return evidence

    for source_name, feed_url in RSS_FEEDS:
        titles = fetch_rss_titles(feed_url, limit=30)
        best = None
        best_sim = 0.0
        for title, link in titles:
            sim = jaccard(q_toks, normalize_tokens(title))
            if sim > best_sim:
                best_sim = sim
                best = (title, link)

        # порог сходства (эмпирически)
        if best and best_sim >= 0.22:
            title, link = best
            evidence.append(EvidenceItem(
                source=source_name,
                title=title,
                url=link,
                score=float(best_sim),
                note="похоже по ключевым словам (RSS)"
            ))

    return evidence


# =========================
# Score fusion
# =========================
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def factcheck_score(items: List[EvidenceItem]) -> Tuple[float, str]:
    if not items:
        return 0.5, "нет данных"

    scored = [it.score for it in items if it.score is not None]
    ratings = " | ".join([(it.rating or "") for it in items if it.rating])

    if scored:
        avg = sum(scored) / len(scored)
        if avg <= 0.30:
            return avg, f"фактчекинг склоняется к ложности ({ratings[:180]})"
        if avg >= 0.70:
            return avg, f"фактчекинг склоняется к правдивости ({ratings[:180]})"
        return avg, f"фактчекинг дал смешанный результат ({ratings[:180]})"

    return 0.5, "фактчек найден, но текстовая оценка не распознана"


def news_score(items: List[EvidenceItem]) -> Tuple[float, str]:
    if not items:
        return 0.5, "нет данных"
    if len(items) >= 2:
        return 0.75, "похожие материалы найдены у нескольких источников"
    return 0.65, "похожий материал найден у одного источника"


def fuse_scores(
    style_fake: float,
    clickbait_fake: float,
    fc_items: List[EvidenceItem],
    news_items: List[EvidenceItem]
) -> Tuple[float, str, List[SignalItem]]:
    # переводим fake->truth компоненты
    rubert_truth = 1.0 - clamp01(style_fake)
    clickbait_truth = 1.0 - clamp01(clickbait_fake)

    fc_val, fc_detail = factcheck_score(fc_items)
    news_val, news_detail = news_score(news_items)

    truth = (
        W_RUBERT * rubert_truth +
        W_CLICKBAIT * clickbait_truth +
        W_FACTCHECK * fc_val +
        W_NEWS * news_val
    )
    truth = clamp01(truth)

    # verdict (осторожно, чтобы не быть "слишком уверенным" без источников)
    if fc_items:
        if truth >= THR_TRUE:
            verdict = "likely_true"
        elif truth <= THR_FALSE:
            verdict = "likely_false"
        else:
            verdict = "not_enough_info"
    else:
        # без фактчека делаем более консервативно
        if truth >= 0.70 and news_items:
            verdict = "likely_true"
        elif truth <= 0.30:
            verdict = "likely_false"
        else:
            verdict = "not_enough_info"

    signals = [
        SignalItem(
            name="style_model",
            value=float(rubert_truth),
            weight=W_RUBERT,
            detail=f"RuBERT: style_fake={style_fake:.3f} => truth_component={rubert_truth:.3f}",
        ),
        SignalItem(
            name="clickbait_heuristics",
            value=float(clickbait_truth),
            weight=W_CLICKBAIT,
            detail=f"Clickbait: fake_score={clickbait_fake:.3f} => truth_component={clickbait_truth:.3f}",
        ),
        SignalItem(
            name="factcheck",
            value=float(fc_val),
            weight=W_FACTCHECK,
            detail=fc_detail,
        ),
        SignalItem(
            name="news_sources",
            value=float(news_val),
            weight=W_NEWS,
            detail=news_detail,
        ),
    ]
    return truth, verdict, signals


# =========================
# Endpoint
# =========================
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.time()
    text = (req.text or "").strip()

    if not text:
        return PredictResponse(
            label="unknown",
            score=0.0,
            verdict="not_enough_info",
            truth_score=0.5,
            signals=[],
            suspicious_fragments=[],
            evidence=[],
            meta={"error": "empty text"},
        )

    # 1) RuBERT стиль по чанкам
    style_fake, top_chunks = style_score_long_text(text)

    # 2) Clickbait / эвристики
    cb_fake, cb_detail = clickbait_fake_score(text)

    # 3) Подозрительные фрагменты (предложения)
    susp = suspicious_sentences(text, top_k=3)

    # 4) Claims + factcheck (опционально)
    claims = extract_claims(text, max_claims=6)
    fc_items: List[EvidenceItem] = []
    if ENABLE_FACTCHECK and FACTCHECK_API_KEY:
        for c in claims:
            fc_items.extend(factcheck_search(c))

    # 5) RSS (опционально)
    news_items: List[EvidenceItem] = rss_verify(text) if ENABLE_RSS else []

    # 6) Объединение сигналов
    truth_score, verdict, signals = fuse_scores(style_fake, cb_fake, fc_items, news_items)

    # 7) label/score (для Android)
    if verdict == "likely_false":
        label = "fake"
        score = float(1.0 - truth_score)
    elif verdict == "likely_true":
        label = "real"
        score = float(truth_score)
    else:
        label = "unknown"          # <-- важное изменение
        score = float(truth_score) # показываем нейтрально "насколько похоже на правду"

    # 8) evidence: сначала factcheck, потом news
    evidence = fc_items + news_items

    meta = {
        "device": str(device),
        "model_dir": MODEL_DIR,
        "len_chars": len(text),
        "num_claims": len(claims),
        "claims_preview": [c[:120] for c in claims],
        "top_chunks": [{"p_fake": float(p), "preview": frag[:140]} for frag, p in top_chunks],
        "clickbait_detail": cb_detail,
        "latency_ms": int((time.time() - t0) * 1000),
        "factcheck_enabled": bool(ENABLE_FACTCHECK and FACTCHECK_API_KEY),
        "rss_enabled": bool(ENABLE_RSS),
    }

    return PredictResponse(
        label=label,
        score=score,
        verdict=verdict,
        truth_score=float(truth_score),
        signals=signals,
        suspicious_fragments=susp,
        evidence=evidence,
        meta=meta,
    )