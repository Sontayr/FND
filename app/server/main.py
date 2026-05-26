from __future__ import annotations

import os
import re
import time
import math
import json
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
ENABLE_LLM = True

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

RSS_FEEDS = [
    ("Lenta", "https://lenta.ru/rss/news"),
    ("Kommersant", "https://www.kommersant.ru/RSS/news.xml"),
    ("Kommersant Main", "https://www.kommersant.ru/rss/main.xml"),
    ("TASS", "https://tass.ru/feed"),
    ("RBC", "https://rssexport.rbc.ru/rbcnews/news/30/full.rss"),
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
    text: str = Field(..., description="Текст новости/сообщения")

    use_ml: bool = True
    use_factcheck: bool = True
    use_news: bool = True
    use_llm: bool = False


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


class LLMResult(BaseModel):
    truth_score: float = 0.5
    verdict: str = "not_enough_info"
    explanation: str = "LLM-анализ не дал результата"


class PredictResponse(BaseModel):

    label: str
    score: float

    verdict: str
    truth_score: float

    signals: List[SignalItem] = []
    suspicious_fragments: List[str] = []
    evidence: List[EvidenceItem] = []

    summary_points: List[str] = []
    llm_explanation: Optional[str] = None

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
RUS_STOPWORDS = {
    "этот", "эта", "это", "эти", "того", "тому", "также", "который", "которая",
    "которые", "которое", "свой", "свои", "свою", "своего", "были", "была",
    "было", "будет", "после", "перед", "из-за", "изза", "около", "среди",
    "новости", "сообщил", "сообщила", "сообщили", "заявил", "заявила",
    "заявили", "рассказал", "рассказала", "данные", "стало", "известно",
    "россии", "россия", "рф", "москва", "сегодня", "вчера"
}


def normalize_tokens(s: str) -> List[str]:
    s = s.lower()
    s = s.replace("ё", "е")
    s = re.sub(r"[^a-zа-я0-9\s-]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()

    toks = []
    for t in s.split(" "):
        t = t.strip("-")
        if len(t) < 4:
            continue
        if t in RUS_STOPWORDS:
            continue
        toks.append(t)

    return toks


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / len(sa | sb)


def weighted_token_score(query: str, document: str) -> float:
    """
    Оценка похожести текста новости и материала из RSS.
    Используем простую интерпретируемую метрику:
    - Jaccard по ключевым словам;
    - доля ключевых слов запроса, найденных в документе.
    """
    q = normalize_tokens(query)
    d = normalize_tokens(document)

    if not q or not d:
        return 0.0

    sq, sd = set(q), set(d)

    j = len(sq & sd) / len(sq | sd)
    recall = len(sq & sd) / len(sq)

    score = 0.45 * j + 0.55 * recall

    return float(max(0.0, min(1.0, score)))


def fetch_rss_items(url: str, limit: int = 40) -> List[Dict[str, str]]:
    """
    Возвращает список RSS-элементов:
    {
        "title": "...",
        "link": "...",
        "description": "..."
    }
    """
    try:
        import requests
        import xml.etree.ElementTree as ET
    except Exception:
        return []

    try:
        r = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": "Mozilla/5.0 FakeNewsDetector/1.0"
            }
        )
        r.raise_for_status()

        xml = r.text.strip()

        if not xml:
            return []

        root = ET.fromstring(xml)

        items = []

        # Обычный RSS: channel/item
        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")

            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            link = link_el.text.strip() if link_el is not None and link_el.text else ""
            desc = desc_el.text.strip() if desc_el is not None and desc_el.text else ""

            # чистим HTML из description
            desc = re.sub(r"<[^>]+>", " ", desc)
            desc = re.sub(r"\s+", " ", desc).strip()

            if title:
                items.append(
                    {
                        "title": title,
                        "link": link,
                        "description": desc
                    }
                )

            if len(items) >= limit:
                break

        # Atom fallback: entry/title/link/summary
        if not items:
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall(".//atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                link_el = entry.find("atom:link", ns)

                title = title_el.text.strip() if title_el is not None and title_el.text else ""
                desc = summary_el.text.strip() if summary_el is not None and summary_el.text else ""
                link = link_el.attrib.get("href", "") if link_el is not None else ""

                desc = re.sub(r"<[^>]+>", " ", desc)
                desc = re.sub(r"\s+", " ", desc).strip()

                if title:
                    items.append(
                        {
                            "title": title,
                            "link": link,
                            "description": desc
                        }
                    )

                if len(items) >= limit:
                    break

        return items

    except Exception:
        return []


def build_news_queries(text: str, claims: Optional[List[str]] = None, max_queries: int = 4) -> List[str]:
    """
    Формирует короткие запросы для проверки по новостным источникам.
    Лучше проверять не весь текст, а отдельные утверждения.
    """
    queries = []

    if claims:
        for c in claims:
            c = c.strip()
            if 30 <= len(c) <= 260:
                queries.append(c)

    sents = split_sentences(text)

    for s in sents:
        s = s.strip()
        if 30 <= len(s) <= 260:
            queries.append(s)

    if not queries and text.strip():
        queries.append(text[:220])

    # убираем дубли
    uniq = []
    seen = set()

    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(q)

    return uniq[:max_queries]


def rss_verify(text: str, claims: Optional[List[str]] = None) -> List[EvidenceItem]:
    if not ENABLE_RSS:
        return []

    queries = build_news_queries(text, claims=claims, max_queries=4)

    if not queries:
        return []

    evidence: List[EvidenceItem] = []

    for source_name, feed_url in RSS_FEEDS:
        items = fetch_rss_items(feed_url, limit=50)

        best_item = None
        best_score = 0.0
        best_query = ""

        for item in items:
            doc = f"{item.get('title', '')}. {item.get('description', '')}"

            for q in queries:
                score = weighted_token_score(q, doc)

                if score > best_score:
                    best_score = score
                    best_item = item
                    best_query = q

        # Порог специально не слишком высокий:
        # RSS содержит короткие заголовки, поэтому точного совпадения часто не будет.
        if best_item and best_score >= 0.18:
            evidence.append(
                EvidenceItem(
                    source=source_name,
                    title=best_item.get("title"),
                    url=best_item.get("link"),
                    score=float(best_score),
                    note=f"найдено сходство с новостным источником; запрос: {best_query[:120]}"
                )
            )

    # сортируем по сходству
    evidence.sort(key=lambda x: x.score or 0.0, reverse=True)

    # убираем дубли по ссылке/заголовку
    uniq = []
    seen = set()

    for it in evidence:
        key = (it.url or "", it.title or "")
        if key not in seen:
            seen.add(key)
            uniq.append(it)

    return uniq[:5]


def extract_json_object(raw: str) -> Optional[dict]:
    if not raw:
        return None

    raw = raw.strip()

    # Убираем markdown-обертки, если модель вернула ```json ... ```
    raw = raw.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    candidate = raw[start:end + 1]

    try:
        return json.loads(candidate)
    except Exception:
        return None


def parse_llm_fallback(raw: str) -> Optional[dict]:
    """
    Запасной парсер на случай, если LLM вернула неидеальный JSON.
    Пытается вытащить truth_score, verdict и explanation из обычного текста.
    """
    if not raw:
        return None

    text = raw.strip()

    result = {}

    score_match = re.search(
        r'"?truth_score"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
        text,
        re.IGNORECASE
    )

    if score_match:
        try:
            result["truth_score"] = float(score_match.group(1))
        except Exception:
            result["truth_score"] = 0.5

    verdict_match = re.search(
        r'"?verdict"?\s*[:=]\s*"?([a-zA-Z_]+)"?',
        text,
        re.IGNORECASE
    )

    if verdict_match:
        verdict = verdict_match.group(1).strip()
        if verdict in ["likely_true", "likely_false", "not_enough_info"]:
            result["verdict"] = verdict

    explanation_match = re.search(
        r'"?explanation"?\s*[:=]\s*"([^"]+)"',
        text,
        re.IGNORECASE | re.DOTALL
    )

    if explanation_match:
        result["explanation"] = explanation_match.group(1).strip()

    if "truth_score" not in result:
        return None

    if "verdict" not in result:
        score = result.get("truth_score", 0.5)
        if score >= 0.65:
            result["verdict"] = "likely_true"
        elif score <= 0.35:
            result["verdict"] = "likely_false"
        else:
            result["verdict"] = "not_enough_info"

    if "explanation" not in result:
        result["explanation"] = "LLM-анализ выявил признаки недостоверности или логического противоречия, но объяснение не было корректно извлечено из ответа модели."

    return result


def llm_analyze_text_openrouter(text: str) -> LLMResult:
    if not ENABLE_LLM or not OPENROUTER_API_KEY:
        return LLMResult(
            truth_score=0.5,
            verdict="not_enough_info",
            explanation="LLM-анализ не выполнен: не задан OPENROUTER_API_KEY."
        )

    try:
        import requests
    except Exception:
        return LLMResult(
            truth_score=0.5,
            verdict="not_enough_info",
            explanation="LLM-анализ не выполнен: библиотека requests недоступна."
        )

    safe_text = text.strip()
    if len(safe_text) > 2000:
        safe_text = safe_text[:2000]

    system_prompt = """
Ты являешься модулем проверки достоверности информации в системе выявления фейковых новостей.

Оцени текст с точки зрения:
- общеизвестных фактов;
- исторических фактов;
- логических противоречий;
- неправдоподобных утверждений.

Важно:
- не выдумывай источники;
- не утверждай, что проверил интернет;
- если данных недостаточно, ставь нейтральную оценку;
- explanation должен быть на русском языке, 2-3 предложения.

Верни строго JSON:
{
  "truth_score": число от 0 до 1,
  "verdict": "likely_true" или "likely_false" или "not_enough_info",
  "explanation": "короткое объяснение"
}
""".strip()

    user_prompt = f'Текст для анализа:\n"""{safe_text}"""'

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "FakeNewsDetector"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 500
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        err = str(e)

        if "429" in err:
            msg = "LLM-анализ временно недоступен: превышен лимит бесплатной модели OpenRouter."
        elif "401" in err or "403" in err:
            msg = "LLM-анализ не выполнен: ошибка доступа к OpenRouter API. Проверьте API-ключ."
        elif "503" in err or "502" in err or "Service Unavailable" in err:
            msg = "LLM-анализ временно недоступен: выбранная модель OpenRouter перегружена или недоступна."
        else:
            msg = "LLM-анализ не выполнен: ошибка обращения к OpenRouter API."

        return LLMResult(
            truth_score=0.5,
            verdict="not_enough_info",
            explanation=msg
        )

    try:
        raw_text = data["choices"][0]["message"]["content"]
        print("OPENROUTER RAW RESPONSE:", repr(raw_text))
    except Exception:
        return LLMResult(
            truth_score=0.5,
            verdict="not_enough_info",
            explanation="LLM-анализ не выполнен: неожиданный формат ответа OpenRouter."
        )

    obj = extract_json_object(raw_text)

    if not obj:
        obj = parse_llm_fallback(raw_text)

    if not obj:
        return LLMResult(
            truth_score=0.5,
            verdict="not_enough_info",
            explanation="LLM-анализ не выполнен: модель вернула ответ в формате, который не удалось разобрать."
        )

    truth_score = obj.get("truth_score", 0.5)
    verdict = obj.get("verdict", "not_enough_info")
    explanation = obj.get("explanation", "LLM-анализ выполнен, но объяснение отсутствует.")

    try:
        truth_score = float(truth_score)
    except Exception:
        truth_score = 0.5

    truth_score = clamp01(truth_score)

    if verdict not in ["likely_true", "likely_false", "not_enough_info"]:
        if truth_score >= 0.65:
            verdict = "likely_true"
        elif truth_score <= 0.35:
            verdict = "likely_false"
        else:
            verdict = "not_enough_info"

    return LLMResult(
        truth_score=truth_score,
        verdict=verdict,
        explanation=str(explanation)[:600]
    )


def llm_analyze_text_router(text: str) -> LLMResult:
    if LLM_PROVIDER == "openrouter":
        return llm_analyze_text_openrouter(text)

    if LLM_PROVIDER == "gemini":
        return llm_analyze_text(text)

    return LLMResult(
        truth_score=0.5,
        verdict="not_enough_info",
        explanation=f"LLM-анализ не выполнен: неизвестный провайдер {LLM_PROVIDER}."
    )


def llm_analyze_text(text: str) -> LLMResult:
    """
    LLM-модуль для проверки общеизвестных, исторических и логических фактов.
    Возвращает truth_score 0..1 и короткое объяснение.
    """

    if not ENABLE_LLM or not GEMINI_API_KEY:
        return LLMResult(
            truth_score=0.5,
            verdict="not_enough_info",
            explanation="LLM-анализ не выполнен: не задан GEMINI_API_KEY."
        )

    try:
        import requests
    except Exception:
        return LLMResult(
            truth_score=0.5,
            verdict="not_enough_info",
            explanation="LLM-анализ не выполнен: библиотека requests недоступна."
        )

    safe_text = text.strip()
    if len(safe_text) > 2000:
        safe_text = safe_text[:2000]

    prompt = f"""
Ты являешься модулем проверки достоверности информации в системе выявления фейковых новостей.

Твоя задача — оценить текст с точки зрения:
1. общеизвестных фактов;
2. исторических фактов;
3. логических противоречий;
4. неправдоподобных утверждений.

Важно:
- не выдумывай источники;
- не утверждай, что проверил интернет;
- если данных недостаточно, ставь нейтральную оценку;
- ответ должен быть коротким;
- explanation должен быть на русском языке, 2-3 предложения.

Верни СТРОГО JSON без Markdown:

{{
  "truth_score": число от 0 до 1,
  "verdict": "likely_true" или "likely_false" или "not_enough_info",
  "explanation": "короткое объяснение"
}}

Текст для анализа:
\"\"\"{safe_text}\"\"\"
""".strip()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 512
        }
    }

    params = {
        "key": GEMINI_API_KEY
    }

    try:
        r = requests.post(url, params=params, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        err = str(e)

        if "503" in err or "Service Unavailable" in err:
            msg = "LLM-анализ временно недоступен: сервис Gemini перегружен или недоступен. Попробуйте повторить запрос позже."
        elif "429" in err:
            msg = "LLM-анализ временно недоступен: превышен лимит запросов к Gemini API."
        elif "403" in err or "401" in err:
            msg = "LLM-анализ не выполнен: ошибка доступа к Gemini API. Проверьте API-ключ."
        else:
            msg = "LLM-анализ не выполнен: ошибка обращения к Gemini API."

        return LLMResult(
            truth_score=0.5,
            verdict="not_enough_info",
            explanation=msg
        )

    try:
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        print("GEMINI RAW RESPONSE FULL:", repr(raw_text))
    except Exception:
        return LLMResult(
            truth_score=0.5,
            verdict="not_enough_info",
            explanation="LLM-анализ не выполнен: неожиданный формат ответа Gemini."
        )

    obj = extract_json_object(raw_text)

    if not obj:
        obj = parse_llm_fallback(raw_text)

    if not obj:
        return LLMResult(
            truth_score=0.5,
            verdict="not_enough_info",
            explanation="LLM-анализ не выполнен: модель вернула ответ в формате, который не удалось разобрать."
        )

    truth_score = obj.get("truth_score", 0.5)
    verdict = obj.get("verdict", "not_enough_info")
    explanation = obj.get("explanation", "LLM-анализ выполнен, но объяснение отсутствует.")

    try:
        truth_score = float(truth_score)
    except Exception:
        truth_score = 0.5

    truth_score = clamp01(truth_score)

    if verdict not in ["likely_true", "likely_false", "not_enough_info"]:
        if truth_score >= 0.65:
            verdict = "likely_true"
        elif truth_score <= 0.35:
            verdict = "likely_false"
        else:
            verdict = "not_enough_info"

    return LLMResult(
        truth_score=truth_score,
        verdict=verdict,
        explanation=str(explanation)[:600]
    )
# =========================
# Score fusion
# =========================
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def build_summary_points(
    style_fake: Optional[float],
    clickbait_fake: Optional[float],
    fc_items: List[EvidenceItem],
    news_items: List[EvidenceItem],
    truth_score: float,
    use_ml: bool = True,
    use_factcheck: bool = True,
    use_news: bool = True,
    use_llm: bool = False,
) -> List[str]:

    points = []

    if use_ml and style_fake is not None:
        if style_fake >= 0.8:
            points.append("ML-модель обнаружила признаки недостоверного текста")
        elif style_fake <= 0.3:
            points.append("ML-модель считает стиль текста похожим на достоверный")

    if use_ml and clickbait_fake is not None and clickbait_fake >= 0.6:
        points.append("Обнаружены признаки кликбейта или эмоциональной манипуляции")

    if use_factcheck:
        if fc_items:
            low_scores = [x.score for x in fc_items if x.score is not None and x.score < 0.4]
            high_scores = [x.score for x in fc_items if x.score is not None and x.score > 0.7]

            if low_scores:
                points.append("Фактчекинг нашел возможное опровержение")
            elif high_scores:
                points.append("Фактчекинг нашел подтверждающие материалы")
        else:
            points.append("Фактчекинг не нашел совпадений по этому утверждению")

    if use_news:
        if news_items:
            if len(news_items) >= 2:
                points.append("Событие упоминается в нескольких новостных источниках")
            else:
                points.append("Найден похожий материал в новостных источниках")
        else:
            points.append("Новостные источники не нашли похожих материалов")

    if truth_score <= 0.35:
        points.append("Общая оценка системы склоняется к недостоверности")
    elif truth_score >= 0.65:
        points.append("Общая оценка системы склоняется к достоверности")

    if not points:
        points.append("Выбранные методы проверки не дали уверенного результата")

    return points[:5]


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

    scores = [it.score for it in items if it.score is not None]

    if not scores:
        if len(items) >= 2:
            return 0.70, "похожие материалы найдены у нескольких источников"
        return 0.62, "похожий материал найден у одного источника"

    best = max(scores)
    avg = sum(scores) / len(scores)

    if len(items) >= 2 and best >= 0.25:
        return 0.78, "похожие материалы найдены у нескольких новостных источников"

    if best >= 0.30:
        return 0.72, "найдено сильное сходство с материалом новостного источника"

    if best >= 0.18:
        return 0.63, "найдено частичное сходство с новостным источником"

    return max(0.5, min(0.6, avg)), "найдены слабые совпадения в новостных источниках"


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
            summary_points=["Введите текст новости для проверки"],
            llm_explanation=None,
            meta={"error": "empty text"},
        )

    # =========================
    # 1) ML-модель и эвристики
    # =========================
    style_fake = None
    top_chunks = []
    cb_fake = None
    cb_detail = "disabled"
    susp = []

    if req.use_ml:
        style_fake, top_chunks = style_score_long_text(text)
        cb_fake, cb_detail = clickbait_fake_score(text)
        susp = suspicious_sentences(text, top_k=3)

    # =========================
    # 2) Claims
    # =========================
    claims = extract_claims(text, max_claims=6)

    # =========================
    # 3) FactCheck
    # =========================
    fc_items: List[EvidenceItem] = []

    if req.use_factcheck and ENABLE_FACTCHECK and FACTCHECK_API_KEY:
        for c in claims:
            fc_items.extend(factcheck_search(c))

    # =========================
    # 4) News / RSS
    # =========================
    news_items: List[EvidenceItem] = []

    if req.use_news and ENABLE_RSS:
        news_items = rss_verify(text, claims=claims)

    # =========================
    # 5) LLM пока отключена
    # =========================
    llm_explanation = None

    llm_result: Optional[LLMResult] = None
    llm_explanation = None

    if req.use_llm:
        llm_result = llm_analyze_text_router(text)
        llm_explanation = llm_result.explanation

    # =========================
    # 6) Объединение сигналов
    # =========================
    truth_score, verdict, signals = fuse_scores(
        style_fake if style_fake is not None else 0.5,
        cb_fake if cb_fake is not None else 0.5,
        fc_items,
        news_items
    )
    if req.use_llm and llm_result is not None:
        other_modules_enabled = bool(req.use_ml or req.use_factcheck or req.use_news)

        if other_modules_enabled:
            truth_score = clamp01(0.75 * truth_score + 0.25 * llm_result.truth_score)
        else:
            truth_score = clamp01(llm_result.truth_score)

        signals.append(
            SignalItem(
                name="llm_analysis",
                value=float(llm_result.truth_score),
                weight=0.25 if other_modules_enabled else 1.0,
                detail=llm_result.explanation,
            )
        )

        if truth_score >= THR_TRUE:
            verdict = "likely_true"
        elif truth_score <= THR_FALSE:
            verdict = "likely_false"
        else:
            verdict = "not_enough_info"

    # =========================
    # 7) Краткий summary для пользователя
    # =========================
    summary_points = build_summary_points(
        style_fake,
        cb_fake,
        fc_items,
        news_items,
        truth_score,
        use_ml=req.use_ml,
        use_factcheck=req.use_factcheck,
        use_news=req.use_news,
        use_llm=req.use_llm,
    )

    if req.use_llm and llm_result is not None:
        if llm_result.truth_score <= 0.35:
            summary_points.append("LLM-анализ обнаружил возможные фактические или логические ошибки")
        elif llm_result.truth_score >= 0.65:
            summary_points.append("LLM-анализ не выявил явных противоречий с общеизвестными фактами")
        else:
            summary_points.append("LLM-анализ не дал уверенного вывода")

    # =========================
    # 8) label/score для Android
    # =========================
    if verdict == "likely_false":
        label = "fake"
        score = float(1.0 - truth_score)
    elif verdict == "likely_true":
        label = "real"
        score = float(truth_score)
    else:
        label = "unknown"
        score = float(truth_score)

    # =========================
    # 9) Evidence
    # =========================
    evidence = fc_items + news_items

    # =========================
    # 10) Meta
    # =========================
    meta = {
        "device": str(device),
        "model_dir": MODEL_DIR,
        "len_chars": len(text),
        "num_claims": len(claims),
        "claims_preview": [c[:120] for c in claims],
        "top_chunks": [
            {"p_fake": float(p), "preview": frag[:140]}
            for frag, p in top_chunks
        ],
        "clickbait_detail": cb_detail,
        "latency_ms": int((time.time() - t0) * 1000),
        "factcheck_enabled": bool(req.use_factcheck and ENABLE_FACTCHECK and FACTCHECK_API_KEY),
        "rss_enabled": bool(req.use_news and ENABLE_RSS),
        "ml_enabled": bool(req.use_ml),
        "llm_enabled": bool(req.use_llm),
        "llm_model": GEMINI_MODEL if req.use_llm else None,
        "llm_provider": LLM_PROVIDER if req.use_llm else None,
        "openrouter_model": OPENROUTER_MODEL if req.use_llm and LLM_PROVIDER == "openrouter" else None,
    }

    return PredictResponse(
        label=label,
        score=score,
        verdict=verdict,
        truth_score=float(truth_score),
        signals=signals,
        suspicious_fragments=susp,
        evidence=evidence,
        summary_points=summary_points,
        llm_explanation=llm_explanation,
        meta=meta,
    )