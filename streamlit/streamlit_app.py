#!/usr/bin/env python3
"""
legal case analyzer + RAG precedent retrieval
rules classify (primary + temporal), model tiebreaker, ollama summarizes + judges
python -m streamlit run streamlit_app.py
"""
import json, os, re, time, tempfile, requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import streamlit as st

st.set_page_config(page_title="case analyzer", page_icon="§", layout="centered")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:ital,wght@0,300;0,400;0,500;1,300;1,400&family=IBM+Plex+Mono:wght@300;400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded');

.stApp { background: #111; }
section[data-testid="stSidebar"] { background: #161616; border-right: 1px solid #252525; }
.block-container { max-width: 740px; padding-top: 3rem; }
#MainMenu, footer { visibility: hidden; }

html, body, .stApp, .stMarkdown, p, label, li {
    font-family: 'IBM Plex Serif', Georgia, serif !important;
    color: #d4d4d4;
}
button span, [data-testid="stBaseButton-headerNoPadding"] span,
[data-testid="baseButton-headerNoPadding"] span,
.stIcon, .material-symbols-rounded,
[data-testid="collapsedControl"] span {
    font-family: 'Material Symbols Rounded', sans-serif !important;
}
code, pre, .stCode { font-family: 'IBM Plex Mono', monospace !important; color: #bbb; }

section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span { color: #bbb !important; }
.stSelectbox label, .stToggle label { font-size: 0.82rem; color: #666 !important; }
.stSpinner > div { color: #888 !important; }

div[data-testid="stFileUploader"] {
    background: #1a1a1a; border: 1px dashed #333; border-radius: 8px;
}
div[data-testid="stFileUploader"] span { color: #ccc !important; }
div[data-testid="stFileUploader"] small { color: #666 !important; }
div[data-testid="stFileUploader"] button {
    background: #e0e0e0 !important; color: #111 !important;
    border: none !important; border-radius: 5px !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] { padding: 8px 12px !important; margin-top: 6px; }
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button {
    background: transparent !important; color: #888 !important; margin-left: 12px; padding: 4px 8px !important;
}

.stExpander { border: 1px solid #252525 !important; border-radius: 6px !important; background: #161616 !important; }
.stExpander summary span { color: #999 !important; }
.stExpander svg { color: #666 !important; }

.site-title { font-family: 'IBM Plex Serif', serif; font-size: 1.7rem; font-weight: 500; color: #eee; letter-spacing: -0.3px; margin: 0 0 0.25rem 0; }
.site-sub { font-size: 0.9rem; color: #666; font-weight: 300; font-style: italic; margin: 0 0 2rem 0; }

.label-tag { display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; font-weight: 500; letter-spacing: 1.5px; padding: 6px 16px; border-radius: 4px; margin: 0.8rem 0 0.5rem; }
.tag-ACTIVE { background: #1a2e1a; color: #6fcf6f; }
.tag-DEAD { background: #2e1a1a; color: #f07070; }
.tag-STAGNANT { background: #2e2510; color: #e0a030; }
.tag-REPETITIVE { background: #1a1f2e; color: #70a0f0; }
.tag-UNCLASSIFIED { background: #1e1e1e; color: #777; }

.conf-badge { display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; font-weight: 500; letter-spacing: 1px; padding: 5px 12px; border-radius: 4px; margin-left: 8px; vertical-align: middle; }
.conf-high { background: #1a2e1a; color: #6fcf6f; }
.conf-medium { background: #2e2510; color: #e0a030; }
.conf-low { background: #2e1a1a; color: #f07070; }
.conf-none { background: #1e1e1e; color: #666; }

.reason-text { font-size: 0.9rem; color: #aaa; line-height: 1.7; margin: 0.4rem 0 1.5rem; }

.j-head { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 500; letter-spacing: 1.5px; text-transform: uppercase; color: #555; margin: 2.2rem 0 0.5rem; padding-bottom: 0.35rem; border-bottom: 1px solid #252525; }
.j-body { font-size: 0.93rem; color: #ccc; line-height: 1.8; margin: 0.2rem 0 0.6rem; }
.j-item { font-size: 0.9rem; color: #bbb; line-height: 1.65; padding: 0.35rem 0 0.35rem 1.1rem; margin: 0.3rem 0; border-left: 2px solid #333; }
.j-item-num { font-family: 'IBM Plex Mono', monospace; color: #555; font-size: 0.78rem; margin-right: 0.5rem; }

.thinline { height: 1px; background: #252525; margin: 2rem 0; }
.meta { font-family: 'IBM Plex Mono', monospace; font-size: 0.73rem; color: #444; margin: 2.5rem 0 1rem; }

.sb-head { font-family: 'IBM Plex Serif', serif; font-size: 1.05rem; font-weight: 500; color: #ddd; margin-bottom: 0.15rem; }
.sb-note { font-size: 0.78rem; color: #555; font-style: italic; margin-bottom: 1.3rem; }
.sb-status { font-family: 'IBM Plex Mono', monospace; font-size: 0.73rem; color: #555; margin: 0.2rem 0; }
.sb-dot { color: #5a5; }
.sb-dot-off { color: #444; }

.prec-card { background: #161616; border: 1px solid #252525; border-radius: 6px; padding: 0.8rem 1rem; margin: 0.5rem 0; }
.prec-name { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 500; color: #ccc; margin-bottom: 0.3rem; }
.prec-score { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #555; float: right; }
.prec-snippet { font-size: 0.82rem; color: #888; line-height: 1.5; }
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# rule-based classifier (primary) + temporal logic
# ═══════════════════════════════════════════════════

VALID_LABELS = {"ACTIVE", "DEAD", "STAGNANT", "REPETITIVE"}

def _compile(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]

REPETITIVE_STRONG = _compile([
    r"res\s+judicata", r"same\s+cause\s+of\s+action",
    r"identical\s+(FIR|complaint|petition|matter|issue)",
    r"multiplicity\s+of\s+proceedings", r"duplicate\s+(filing|petition|complaint|case|FIR)",
    r"already\s+(been\s+)?adjudicated", r"same\s+FIR", r"same\s+subject\s*matter",
    r"order\s+II\s+rule\s+2", r"section\s+11\s+(of\s+)?(the\s+)?CPC",
    r"abuse\s+of\s+(the\s+)?process",
])
REPETITIVE_MODERATE = _compile([
    r"companion\s+(case|petition|matter|appeal)", r"connected\s+(case|petition|matter|appeal)",
    r"clubbed\s+(with|together)", r"consolidated\s+(with|hearing)",
    r"tagged\s+(with|along)", r"lead\s+(case|matter|petition)",
    r"common\s+(question|issue|subject)",
])
STAGNANT_STRONG = _compile([
    r"adjourned\s+sine\s+die", r"kept\s+pending\s+indefinitely",
    r"no\s+(further\s+)?progress", r"languishing", r"dormant", r"no\s+effective\s+hearing",
])
STAGNANT_MODERATE = _compile([
    r"none\s+(appeared|present)\s+for\s+(the\s+)?(petitioner|appellant|respondent)",
    r"(petitioner|appellant)\s+(has\s+)?(not\s+)?(appeared|shown\s+interest)",
    r"no\s+steps?\s+(taken|have\s+been\s+taken)",
    r"matter\s+(has\s+been|was)\s+adjourned\s+(repeatedly|several\s+times)",
    r"not\s+been\s+prosecuted", r"want\s+of\s+prosecution",
])
ACTIVE_STRONG = _compile([
    r"remand(ed)?\s+(back\s+)?(to|for)", r"remit(ted)?\s+(back\s+)?(to|for)",
    r"(sent|send)\s+back\s+(to|for)", r"matter\s+(is\s+)?remanded",
    r"fresh\s+(hearing|consideration|adjudication|decision|trial|inquiry|investigation)",
    r"de\s+novo\s+(trial|hearing|consideration)",
    r"direct(ed|s)?\s+(the\s+)?(trial\s+court|high\s+court|tribunal|lower\s+court|authority)\s+to",
    r"liberty\s+(is\s+)?(granted|given|reserved)\s+to",
    r"liberty\s+to\s+(file|apply|approach|move)",
    r"next\s+date\s+of\s+hearing", r"listed?\s+(for|on)\s+\d",
    r"posted\s+for\s+(hearing|further\s+hearing|orders)",
    r"part(ly|ial(ly)?)\s+(allowed|disposed)", r"partially\s+set\s+aside",
])
ACTIVE_MODERATE = _compile([
    r"pending\s+(appeal|application|petition|review|reference|complaint)",
    r"appeal\s+(is\s+)?(still\s+)?pending", r"subject\s+to\s+(the\s+)?outcome",
    r"further\s+(orders?|proceedings?|inquiry|investigation|hearing)",
    r"shall\s+(be\s+)?(heard|considered|decided|taken\s+up)",
    r"interim\s+(order|relief|stay|injunction)", r"status\s+quo",
    r"stay\s+(granted|continued|extended)", r"without\s+prejudice\s+to",
    r"at\s+liberty\s+to",
])
DEAD_STRONG = _compile([
    r"(appeal|petition|application|complaint|case|SLP|writ\s+petition)\s+(is\s+)?(hereby\s+)?(dismissed|rejected|disposed\s+of)",
    r"(conviction|sentence)\s+(is\s+)?(hereby\s+)?(confirmed|upheld|affirmed|maintained)",
    r"(acquittal|order)\s+(is\s+)?(hereby\s+)?(confirmed|upheld|affirmed|maintained)",
    r"(appeal|petition|SLP)\s+(is\s+)?(hereby\s+)?allowed",
    r"petition\s+stands?\s+(disposed|closed|rejected)",
    r"case\s+stands?\s+closed",
    r"no\s+merit\s+in\s+(this|the)\s+(appeal|petition|application|SLP)",
    r"we\s+find\s+no\s+(merit|substance|force|reason)",
    r"decree\s+(is\s+)?(hereby\s+)?(passed|granted|made)",
    r"suit\s+(is\s+)?(hereby\s+)?(decreed|dismissed)",
    r"finally\s+(disposed|decided|adjudicated|settled)",
    r"nothing\s+survives?\s+in\s+(this|the)",
])
DEAD_MODERATE = _compile([
    r"accordingly\s+(disposed|dismissed|rejected|allowed)",
    r"no\s+(further\s+)?orders?\s+(are\s+)?(necessary|needed|required|called\s+for)",
    r"disposed\s+of\s+in\s+(the\s+)?above\s+terms",
    r"impugned\s+(order|judgment|decree)\s+(is\s+)?(set\s+aside|upheld|affirmed|quashed)",
])


# ── Temporal analysis ──

DATE_PATTERNS = [
    re.compile(r'(\d{1,2})[./\-](\d{1,2})[./\-](20\d{2})'),
    re.compile(r'(\d{1,2})[./\-](\d{1,2})[./\-](19\d{2})'),
    re.compile(r'(\d{1,2})\s+(?:of\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)[,\s]+(\d{4})', re.IGNORECASE),
    re.compile(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})[,\s]+(\d{4})', re.IGNORECASE),
    re.compile(r'(\d{1,2})\s+(?:of\s+)?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[.,\s]+(\d{4})', re.IGNORECASE),
    re.compile(r'dated\s+(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})', re.IGNORECASE),
]

MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}

STALE_ACTIVE_YEARS = 3
STALE_TO_DEAD_YEARS = 7
RECENT_THRESHOLD_YEARS = 2


def extract_dates(text):
    """Extract all dates from text."""
    dates = []
    for pat in DATE_PATTERNS:
        for m in pat.finditer(text):
            groups = m.groups()
            try:
                a, b, c = groups
                if b.isdigit() and a.isdigit():
                    day, month, year = int(a), int(b), int(c)
                    if year < 100:
                        year += 2000 if year < 50 else 1900
                elif b.lower() in MONTHS:
                    day, month, year = int(a), MONTHS[b.lower()], int(c)
                elif a.lower() in MONTHS:
                    month, day, year = MONTHS[a.lower()], int(b), int(c)
                else:
                    continue
                if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2030:
                    dates.append(datetime(year, month, min(day, 28)))
            except (ValueError, TypeError):
                continue
    return sorted(set(dates))


def temporal_analysis(text):
    """Analyze temporal signals in judgment text."""
    dates = extract_dates(text)
    now = datetime.now()

    if not dates:
        return {"has_dates": False, "years_since_last": None, "most_recent": None, "oldest": None}

    most_recent = max(dates)
    oldest = min(dates)
    years_since = (now - most_recent).days / 365.25

    return {
        "has_dates": True,
        "years_since_last": round(years_since, 1),
        "most_recent": most_recent.strftime("%d %b %Y"),
        "oldest": oldest.strftime("%d %b %Y"),
        "date_count": len(dates),
    }


def classify_rules(text):
    """Rule-based classification with temporal adjustment.
    Returns (label or None, confidence, scores, temporal, temporal_note)."""
    t = " ".join(text.split())
    scores = {"REPETITIVE": 0, "STAGNANT": 0, "ACTIVE": 0, "DEAD": 0}

    for p in REPETITIVE_STRONG:
        if p.search(t): scores["REPETITIVE"] += 3
    for p in REPETITIVE_MODERATE:
        if p.search(t): scores["REPETITIVE"] += 1
    for p in STAGNANT_STRONG:
        if p.search(t): scores["STAGNANT"] += 3
    for p in STAGNANT_MODERATE:
        if p.search(t): scores["STAGNANT"] += 1
    for p in ACTIVE_STRONG:
        if p.search(t): scores["ACTIVE"] += 3
    for p in ACTIVE_MODERATE:
        if p.search(t): scores["ACTIVE"] += 1
    for p in DEAD_STRONG:
        if p.search(t): scores["DEAD"] += 3
    for p in DEAD_MODERATE:
        if p.search(t): scores["DEAD"] += 1

    # ── Initial label from pattern scores ──
    label = None
    conf = "low"

    if scores["REPETITIVE"] >= 3:
        label = "REPETITIVE"
        conf = "high" if scores["REPETITIVE"] >= 6 else "medium"
    elif scores["STAGNANT"] >= 3 and scores["ACTIVE"] <= scores["STAGNANT"]:
        label = "STAGNANT"
        conf = "high" if scores["STAGNANT"] >= 6 else "medium"
    elif scores["ACTIVE"] >= 3:
        active_strong = sum(1 for p in ACTIVE_STRONG if p.search(t))
        if active_strong >= 1:
            label = "ACTIVE"
            conf = "high" if scores["ACTIVE"] >= 6 else "medium"
        elif scores["ACTIVE"] > scores["DEAD"]:
            label = "ACTIVE"
            conf = "medium" if scores["ACTIVE"] >= 4 else "low"
    elif scores["DEAD"] >= 3:
        label = "DEAD"
        conf = "high" if scores["DEAD"] >= 6 else "medium"

    # ── Temporal adjustment ──
    temporal = temporal_analysis(text)
    temporal_note = ""

    if temporal["has_dates"] and temporal["years_since_last"] is not None:
        yrs = temporal["years_since_last"]

        # ACTIVE but last activity was long ago → STAGNANT
        if label == "ACTIVE" and yrs >= STALE_ACTIVE_YEARS:
            label = "STAGNANT"
            conf = "medium"
            temporal_note = f"Reclassified: active signals found but last activity was {yrs:.0f} years ago — case appears stagnant."

        # ACTIVE with recent dates → confirm ACTIVE with high confidence
        elif label == "ACTIVE" and yrs <= RECENT_THRESHOLD_YEARS:
            conf = "high"
            temporal_note = f"Confirmed: active proceedings with recent activity ({temporal['most_recent']})."

        # STAGNANT for too long → effectively DEAD
        elif label == "STAGNANT" and yrs >= STALE_TO_DEAD_YEARS:
            label = "DEAD"
            conf = "medium"
            temporal_note = f"Reclassified: stagnant for {yrs:.0f} years with no movement — case is effectively dead."

        # STAGNANT with moderate time → confirm STAGNANT
        elif label == "STAGNANT" and STALE_ACTIVE_YEARS <= yrs < STALE_TO_DEAD_YEARS:
            temporal_note = f"Case stagnant since {temporal['most_recent']} ({yrs:.0f} years)."

        # No label but case is very old → likely DEAD
        elif label is None and yrs >= STALE_TO_DEAD_YEARS:
            label = "DEAD"
            conf = "low"
            temporal_note = f"No strong classification signals, but last activity was {yrs:.0f} years ago."

        # DEAD with recent dates → confirm with high confidence
        elif label == "DEAD" and yrs <= 1:
            conf = "high"
            temporal_note = f"Recently disposed ({temporal['most_recent']})."

    return label, conf, scores, temporal, temporal_note


def build_rule_reason(label, scores, temporal, temporal_note):
    """Build explanation of classification decision."""
    parts = []

    if label == "DEAD":
        if scores["DEAD"] >= 6: parts.append("Strong disposal/dismissal language detected in judgment")
        elif scores["DEAD"] >= 3: parts.append("Disposal signals found in judgment text")
        if scores["ACTIVE"] > 0: parts.append(f"some active signals present (score {scores['ACTIVE']}) but outweighed by disposal language")
    elif label == "ACTIVE":
        if scores["ACTIVE"] >= 6: parts.append("Strong remand/pending/continuation signals detected")
        elif scores["ACTIVE"] >= 3: parts.append("Active proceedings signals found in text")
        if scores["DEAD"] > 0: parts.append(f"disposal language present (score {scores['DEAD']}) but overridden by active signals")
    elif label == "STAGNANT":
        parts.append("Case shows signs of prolonged inactivity or repeated adjournments")
    elif label == "REPETITIVE":
        parts.append("Duplicate filing or same cause of action signals detected")

    if temporal_note:
        parts.append(temporal_note)
    elif temporal.get("has_dates") and temporal.get("most_recent"):
        parts.append(f"Last recorded activity: {temporal['most_recent']}")

    return ". ".join(parts) + "." if parts else ""


# ═══════════════════════════════════════════════════
# backend — model calls
# ═══════════════════════════════════════════════════

FT_SYS = (
    "You are a legal AI assistant specializing in Indian Supreme Court judgments. "
    "Given the text of a court judgment, provide:\n"
    "1. SUMMARY: A concise abstractive summary of the case (key facts, legal issues, and ruling)\n"
    "2. CASE TYPE: Classify as exactly one of: Active | Repetitive | Stagnant | Dead\n"
    "3. CONFIDENCE: Your confidence in the classification: high | medium | low\n\n"
    "Definitions:\n"
    "- Active: Case has ongoing proceedings, pending appeals, or unresolved matters.\n"
    "- Dead: Final judgment delivered, no pending appeals, case fully and finally disposed.\n"
    "- Stagnant: Case adjourned sine die or indefinitely with no substantive progress.\n"
    "- Repetitive: Case involves duplicate filings, same FIR or cause of action as another case.\n\n"
    "Format your response exactly as:\n"
    "SUMMARY: <your summary>\n"
    "CASE TYPE: <Active|Repetitive|Stagnant|Dead>\n"
    "CONFIDENCE: <high|medium|low>"
)

SUMM_SYS = (
    "You are a legal summarizer. Given the text of an Indian court judgment, write a concise "
    "summary in 3-5 sentences covering: the parties involved, the key legal issue, the statutes "
    "or provisions cited, and the final outcome/order. Be precise and factual. "
    "Do not include any labels or classifications — just the summary text."
)

JDG_SYS = ("You are a senior Indian legal analyst. Analyze this case and respond with ONLY a valid JSON object. "
    "No markdown, no explanation, no text outside the JSON. Use this exact structure:\n\n"
    '{"case_summary": "3-5 sentences summarizing the case",'
    ' "facts": "3-5 sentences covering key facts",'
    ' "legal_issues": ["issue 1", "issue 2"],'
    ' "judgment_reasoning": "5-8 sentences referencing law",'
    ' "verdict_recommendation": "suggest what the verdict could be, phrased as a recommendation — use language like the court may consider, it is suggested that, based on the analysis it appears, etc. No statutes or precedents here",'
    ' "statutes_cited": ["statute 1", "statute 2"],'
    ' "precedents_cited": ["case name 1 (year)", "case name 2 (year)"]}\n\n'
    "IMPORTANT: statutes_cited must ONLY contain statutes/sections of law. "
    "precedents_cited must ONLY contain case names. "
    "Do NOT repeat statutes or precedents inside other fields.")

JDG_SYS_WITH_RAG = ("You are a senior Indian legal analyst. You are given a case to analyze, "
    "along with RELATED PRECEDENTS retrieved from a knowledge base. Use the precedents to "
    "inform your reasoning where relevant.\n\n"
    "Respond with ONLY a valid JSON object. No markdown, no explanation, no text outside the JSON. "
    "Use this exact structure:\n\n"
    '{"case_summary": "3-5 sentences summarizing the case",'
    ' "facts": "3-5 sentences covering key facts",'
    ' "legal_issues": ["issue 1", "issue 2"],'
    ' "judgment_reasoning": "5-8 sentences referencing law and precedents",'
    ' "verdict_recommendation": "suggest what the verdict could be, phrased as a recommendation — use language like the court may consider, it is suggested that, based on the analysis it appears, etc. No statutes or precedents here",'
    ' "statutes_cited": ["statute 1", "statute 2"],'
    ' "precedents_cited": ["case name 1 (year)", "case name 2 (year)"]}\n\n'
    "IMPORTANT: statutes_cited must ONLY contain statutes/sections of law. "
    "precedents_cited must ONLY contain case names (include any from the retrieved precedents). "
    "Do NOT repeat statutes or precedents inside other fields.")


def extract_pdf(f):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(f.getvalue())
        p = tmp.name
    t = ""
    try:
        import pdfplumber
        with pdfplumber.open(p) as pdf:
            for pg in pdf.pages:
                x = pg.extract_text()
                if x: t += x + "\n\n"
    except: pass
    if len(t) < 50:
        try:
            from pypdf import PdfReader
            for pg in PdfReader(p).pages:
                x = pg.extract_text()
                if x: t += x + "\n\n"
        except: pass
    os.unlink(p)
    t = re.sub(r'Indian Kanoon\s*-\s*http://indiankanoon\.org/\S*\s*\d*', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def prep(t, h=1500, tl=1000):
    return t if len(t) <= h + tl else t[:h] + "\n[...]\n" + t[-tl:]


# ── llama.cpp (fine-tuned — tiebreaker only) ──

LLAMA_URL = "http://localhost:8080/v1/chat/completions"

def llama_ok():
    try:
        r = requests.get("http://localhost:8080/health", timeout=3)
        return r.status_code == 200
    except: return False

def classify_llama(text):
    truncated = prep(text, 800, 700)
    try:
        r = requests.post(LLAMA_URL, json={
            "model": "local", "temperature": 0.3, "max_tokens": 512, "repeat_penalty": 1.3,
            "messages": [
                {"role": "system", "content": FT_SYS},
                {"role": "user", "content": f"/no_think\nJudgment:\n{truncated}"},
            ],
        }, timeout=120)
        if r.status_code != 200:
            return "UNCLASSIFIED", ""
        output = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return "UNCLASSIFIED", ""

    case_type = "UNCLASSIFIED"
    cm = re.search(r"CASE TYPE:\s*(\w+)", output)
    if cm:
        ct = cm.group(1).strip()
        if ct.upper() in VALID_LABELS: case_type = ct.upper()
    return case_type, ""


# ── ollama (summary + judgment) ──

def ollama_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200: return [m['name'] for m in r.json().get('models', [])]
    except: pass
    return []

def ollama_call(model, sys, usr, np=1500):
    try:
        r = requests.post("http://localhost:11434/api/chat", json={
            "model": model, "stream": False, "think": False,
            "messages": [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            "options": {"temperature": 0.1, "num_predict": np},
        }, timeout=180)
        if r.status_code == 200: return r.json()['message']['content'].strip()
    except: pass
    return None

def summarize_ol(model, text):
    r = ollama_call(model, SUMM_SYS, f"Summarize this judgment:\n\n{prep(text, 2000, 1500)}", 300)
    if not r: return ""
    c = re.sub(r'<think>.*?</think>', '', r, flags=re.DOTALL).strip()
    return c if c else r

def plabel(r):
    c = re.sub(r'<think>.*?</think>', '', r, flags=re.DOTALL).strip()
    if not c: c = r
    u = c.upper()
    w = re.sub(r'[^A-Z]', '', u.split()[0]) if u.split() else ""
    if w in VALID_LABELS: return w, c
    for l in VALID_LABELS:
        if l in u: return l, c
    return "UNCLASSIFIED", c

def classify_ol(model, text):
    CLS_SYS = ("You are a legal case classifier. Respond with exactly one word: "
        "ACTIVE, DEAD, STAGNANT, or REPETITIVE. Then a short reason. Start with the label.")
    r = ollama_call(model, CLS_SYS, f"Classify. First word = label.\n\n{prep(text)}", 100)
    if not r: return "UNCLASSIFIED", ""
    l, c = plabel(r)
    return l, ""

def judge_ol(model, text, precedents=None):
    if precedents:
        prec_text = "\n\n".join([
            f"PRECEDENT {i+1}: {p['source_file']}\n{p['snippet']}"
            for i, p in enumerate(precedents[:3])
        ])
        sys_prompt = JDG_SYS_WITH_RAG
        user_msg = f"RELATED PRECEDENTS:\n{prec_text}\n\nCASE TO ANALYZE:\n{prep(text, 2000, 1500)}"
    else:
        sys_prompt = JDG_SYS
        user_msg = f"Analyze:\n\n{prep(text, 2000, 1500)}"
    r = ollama_call(model, sys_prompt, user_msg, 1500)
    if not r: return {"error": "no response"}
    return parse_judgment(r)


# ── gemini ──

def gemini_ok():
    try:
        from dotenv import load_dotenv; load_dotenv()
    except: pass
    return bool(os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'))

def gcall(prompt):
    try:
        from google import genai; from dotenv import load_dotenv; load_dotenv()
        k = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        return genai.Client(api_key=k).models.generate_content(model="gemini-2.5-flash", contents=prompt).text.strip()
    except: return None

def classify_g(text):
    r = gcall(f'Classify as ACTIVE/DEAD/STAGNANT/REPETITIVE. JSON: {{"label":"X","reason":"..."}}\n\n{prep(text, 3000, 2000)}')
    if not r: return "UNCLASSIFIED", ""
    try:
        j = json.loads(re.sub(r'^```json\s*|```$', '', r).strip())
        l = str(j.get('label', '')).upper()
        if l in VALID_LABELS: return l, ""
    except: pass
    for l in VALID_LABELS:
        if l in r.upper(): return l, ""
    return "UNCLASSIFIED", ""

def summarize_g(text):
    r = gcall(f"{SUMM_SYS}\n\nJudgment:\n{prep(text, 3000, 2000)}")
    return r if r else ""

def judge_g(text, precedents=None):
    if len(text) > 80000: text = text[:40000] + "\n[...]\n" + text[-40000:]
    if precedents:
        prec_text = "\n\n".join([f"PRECEDENT {i+1}: {p['source_file']}\n{p['snippet']}" for i, p in enumerate(precedents[:3])])
        r = gcall(f"{JDG_SYS_WITH_RAG}\n\nRELATED PRECEDENTS:\n{prec_text}\n\nCASE TO ANALYZE:\n{text}")
    else:
        r = gcall(f"{JDG_SYS}\n\nCase:\n{text}")
    if not r: return {"error": "gemini didn't respond"}
    return parse_judgment(r)


# ── RAG ──

def rag_available():
    try:
        from rag_engine import get_index_stats
        stats = get_index_stats()
        return stats.get("indexed", False)
    except: return False

def rag_search(text, exclude_file=None):
    try:
        from rag_engine import search
        return search(text, exclude_file=exclude_file)
    except: return []

def rag_stats():
    try:
        from rag_engine import get_index_stats
        return get_index_stats()
    except: return {"indexed": False, "num_chunks": 0, "num_docs": 0}


# ── parsing ──

def parse_judgment(r):
    c = re.sub(r'<think>.*?</think>', '', r, flags=re.DOTALL).strip()
    if not c: c = r

    # ── try JSON parse first ──
    cleaned = re.sub(r'^```(?:json)?\s*', '', c, flags=re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            valid_keys = {"case_summary", "facts", "legal_issues", "judgment_reasoning",
                          "verdict_recommendation", "statutes_cited", "precedents_cited"}
            res = {}
            for k in valid_keys:
                v = obj.get(k)
                if not v: continue
                if isinstance(v, list):
                    res[k] = "\n".join(f"{i+1}. {str(item)}" for i, item in enumerate(v))
                else:
                    res[k] = str(v)
            if res:
                return res
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # ── fallback: regex section parsing (for non-JSON model output) ──
    _HEADERS = r'(CASE SUMMARY|FACTS|LEGAL ISSUES|JUDGMENT REASONING|VERDICT RECOMMENDATION|STATUTES CITED|PRECEDENTS CITED)'
    c = re.sub(rf'\*\*\s*{_HEADERS}\s*:?\s*\*\*:?', r'\1:', c, flags=re.IGNORECASE)
    c = re.sub(rf'^#{1,4}\s*{_HEADERS}\s*:?\s*$', r'\1:', c, flags=re.IGNORECASE | re.MULTILINE)
    c = re.sub(rf'__\s*{_HEADERS}\s*:?\s*__:?', r'\1:', c, flags=re.IGNORECASE)

    sm = {"CASE SUMMARY": "case_summary", "FACTS": "facts", "LEGAL ISSUES": "legal_issues",
          "JUDGMENT REASONING": "judgment_reasoning", "VERDICT RECOMMENDATION": "verdict_recommendation",
          "STATUTES CITED": "statutes_cited", "PRECEDENTS CITED": "precedents_cited"}
    headers_pat = r'(?:CASE SUMMARY|FACTS|LEGAL ISSUES|JUDGMENT REASONING|VERDICT RECOMMENDATION|STATUTES CITED|PRECEDENTS CITED)'

    res = {}
    for h, k in sm.items():
        m = re.search(rf'{h}\s*:\s*\n?(.*?)(?=\n\s*{headers_pat}\s*:|$)',
                       c, re.DOTALL | re.IGNORECASE)
        if m:
            val = m.group(1).strip().strip('*').strip()
            val = re.sub(r'^\*\*\s*', '', val)
            val = re.sub(rf'(?:^|\n)\s*(?:#{1,4}\s+)?(?:\*\*\s*)?{headers_pat}\s*:?\s*(?:\*\*)?\s*', '\n', val, flags=re.IGNORECASE).strip()
            if val: res[k] = val
    if not res:
        if len(c) > 50: return {"case_summary": c[:2000], "_note": "raw model output"}
        return {"error": "empty response"}

    # fallback post-processing
    prose_fields = {"case_summary", "facts", "judgment_reasoning", "verdict_recommendation"}
    for k in prose_fields:
        if k not in res: continue
        m = re.search(r'(?:^|\.\s+)(\d+\.\s+(?:Section\s+\d|Article\s+\d|Part[\s-]|Exception\s+\d|The\s+\w+.*?\bAct\b|[A-Z][\w\s]+\bvs?\.?\b))', res[k], re.IGNORECASE)
        if m:
            truncated = res[k][:m.start()].rstrip().rstrip('.')
            if len(truncated) > 50:
                res[k] = truncated + '.'

    if "statutes_cited" in res:
        from_split = split_items(res["statutes_cited"])
        real_statutes = [i for i in from_split if not re.search(r'\bvs?\.?\s', i, re.IGNORECASE)]
        stray_prec = [i for i in from_split if re.search(r'\bvs?\.?\s', i, re.IGNORECASE)]
        if stray_prec:
            res["statutes_cited"] = "\n".join(f"{n+1}. {s}" for n, s in enumerate(real_statutes)) if real_statutes else ""
            if "precedents_cited" not in res or not res["precedents_cited"]:
                res["precedents_cited"] = "\n".join(f"{n+1}. {p}" for n, p in enumerate(stray_prec))
            if not res["statutes_cited"]:
                del res["statutes_cited"]

    return res

def split_items(text):
    if isinstance(text, list):
        return [re.sub(r'^\d+\.\s*', '', str(x)).strip().strip('*') for x in text]
    text = str(text).strip()
    # remove any stray section headers (any format) before splitting
    text = re.sub(r'(?:^|\n)\s*(?:#{1,4}\s+)?(?:\*\*\s*)?(?:CASE SUMMARY|FACTS|LEGAL ISSUES|JUDGMENT REASONING|VERDICT RECOMMENDATION|STATUTES CITED|PRECEDENTS CITED)\s*:?\s*(?:\*\*)?\s*', '\n', text, flags=re.IGNORECASE).strip()
    parts = re.split(r'(?:^|\s)(\d+)\.\s+', text)
    items = []
    i = 1
    while i < len(parts) - 1:
        items.append(parts[i + 1].strip())
        i += 2
    if not items:
        if ';' in text: items = [x.strip() for x in text.split(';') if x.strip()]
        elif '\n' in text: items = [x.strip() for x in text.split('\n') if x.strip()]
        else: items = [text]
    cleaned = []
    for item in items:
        item = re.sub(r'^\d+\.\s*', '', item).strip('*').strip().rstrip(',').strip()
        if item: cleaned.append(item)
    return cleaned if cleaned else [text.strip('*')]

def clean_text(text):
    if isinstance(text, list): return [clean_text(t) for t in text]
    text = str(text).strip('*').strip()
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'^\*\*\s*', '', text)
    text = re.sub(r'\s*\*\*$', '', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # strip stray section headers in any format (bold, ##, plain)
    text = re.sub(r'(?:^|\n)\s*(?:#{1,4}\s+)?(?:\*\*\s*)?(?:CASE SUMMARY|FACTS|LEGAL ISSUES|JUDGMENT REASONING|VERDICT RECOMMENDATION|STATUTES CITED|PRECEDENTS CITED)\s*:?\s*(?:\*\*)?\s*', '\n', text, flags=re.IGNORECASE)
    return text.strip()


# ═══════════════════════════════════════════════════
# app
# ═══════════════════════════════════════════════════

# ── auto-detect backends (no UI — using current config) ──
lm = llama_ok()
om = ollama_models()
gm = gemini_ok()
rag = rag_available()
backends = []
if lm: backends.append("legal-classifier:latest")
if om:
    for m in om:
        if m not in backends: backends.append(m)
if gm: backends.append("gemini")
backend = backends[0] if backends else None


st.markdown('<div class="site-title">Legal Case Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="site-sub">upload a court pdf. get a classification and judgment.</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("pdf", type=["pdf"], label_visibility="collapsed")

if not uploaded:
    st.markdown("""
    <div style="margin-top:2.5rem; font-size:0.9rem; color:#888; line-height:1.9;">
    drop a legal case pdf above.<br>
    the tool extracts text, classifies the case as
    <span style="color:#6fcf6f; font-weight:500">active</span>,
    <span style="color:#f07070; font-weight:500">dead</span>,
    <span style="color:#e0a030; font-weight:500">stagnant</span>, or
    <span style="color:#70a0f0; font-weight:500">repetitive</span>,<br>
    then generates a detailed judgment for active cases.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="meta">built for indian legal cases · no data leaves your machine</div>', unsafe_allow_html=True)
    st.stop()

if not backends:
    st.error("no backend available")
    st.stop()


# ── extract ──
with st.spinner("reading pdf..."):
    text = extract_pdf(uploaded)

if len(text) < 50:
    st.error("couldn't extract text from this pdf. might be scanned/image-only.")
    st.stop()

st.markdown(f'<div class="meta">extracted {len(text):,} characters from {uploaded.name}</div>', unsafe_allow_html=True)

with st.expander("view extracted text"):
    st.text(text[:4000] + ("\n..." if len(text) > 4000 else ""))


# ── classify (rules + temporal primary, model tiebreaker) ──
st.markdown('<div class="thinline"></div>', unsafe_allow_html=True)

with st.spinner("classifying..."):
    t0 = time.time()

    # Step 1: rules + temporal (primary)
    rule_label, rule_conf, rule_scores, temporal, temporal_note = classify_rules(text)

    # Step 2: model tiebreaker (only if rules uncertain)
    if rule_label:
        label = rule_label
        confidence = rule_conf
    else:
        if backend == "legal-classifier:latest":
            model_label, _ = classify_llama(text)
        elif backend == "gemini":
            model_label, _ = classify_g(text)
        else:
            model_label, _ = classify_ol(backend, text)

        if model_label != "UNCLASSIFIED":
            label = model_label
            confidence = "low"
        else:
            label = "DEAD"
            confidence = "low"

    cls_time = time.time() - t0

# Build reason from rules + temporal
reason = build_rule_reason(label, rule_scores, temporal, temporal_note)

# Get summary + RAG precedents in parallel
summary = ""
precedents = []

def _summarize():
    if om: return summarize_ol(om[0], text)
    elif gm: return summarize_g(text)
    return ""

def _rag():
    if rag: return rag_search(text, exclude_file=uploaded.name)
    return []

with st.spinner("analyzing..."):
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_summary = pool.submit(_summarize)
        fut_rag = pool.submit(_rag)
        summary = fut_summary.result()
        precedents = fut_rag.result()

# Display classification
conf_class = f"conf-{confidence}" if confidence in ("high", "medium", "low") else "conf-none"
conf_html = f'<span class="conf-badge {conf_class}">{confidence}</span>' if confidence else ""
st.markdown(f'<div class="label-tag tag-{label}">{label}</div>{conf_html}', unsafe_allow_html=True)

# Display summary (ollama) or rule reason as fallback
if summary:
    st.markdown(f'<div class="reason-text">{clean_text(summary)}</div>', unsafe_allow_html=True)
elif reason:
    st.markdown(f'<div class="reason-text">{reason}</div>', unsafe_allow_html=True)


# ── RAG: display precedents ──
if precedents:
    st.markdown('<div class="j-head">related precedents</div>', unsafe_allow_html=True)
    for p in precedents:
        score_pct = int(p["score"] * 100)
        st.markdown(f'''<div class="prec-card">
            <div class="prec-name">{p["source_file"]}<span class="prec-score">{score_pct}% match</span></div>
            <div class="prec-snippet">{p["snippet"][:250]}...</div>
        </div>''', unsafe_allow_html=True)


# ── judgment (active cases only) ──
jdg_time = 0
if label == "ACTIVE":
    st.markdown('<div class="thinline"></div>', unsafe_allow_html=True)

    if backend == "legal-classifier:latest":
        if om:
            jdg_model = om[0]
            with st.spinner("generating judgment — this takes a minute..."):
                t0 = time.time()
                judgment = judge_ol(jdg_model, text, precedents if precedents else None)
                jdg_time = time.time() - t0
        elif gm:
            with st.spinner("generating judgment..."):
                t0 = time.time()
                judgment = judge_g(text, precedents if precedents else None)
                jdg_time = time.time() - t0
        else:
            judgment = None
            st.markdown('<div class="reason-text">no judgment backend available.</div>', unsafe_allow_html=True)
    elif backend == "gemini":
        with st.spinner("generating judgment..."):
            t0 = time.time()
            judgment = judge_g(text, precedents if precedents else None)
            jdg_time = time.time() - t0
    else:
        with st.spinner("generating judgment — this takes a minute..."):
            t0 = time.time()
            judgment = judge_ol(backend, text, precedents if precedents else None)
            jdg_time = time.time() - t0

    if judgment:
        if 'error' in judgment:
            st.markdown(f'<div class="reason-text">judgment failed: {judgment.get("error","unknown")}</div>', unsafe_allow_html=True)
        else:
            if '_note' in judgment: st.caption(f"note: {judgment['_note']}")
            sections = [
                ("case summary", "case_summary"), ("facts", "facts"),
                ("legal issues", "legal_issues"), ("judgment reasoning", "judgment_reasoning"),
                ("suggested verdict", "verdict_recommendation"),
                ("statutes cited", "statutes_cited"), ("precedents cited", "precedents_cited"),
            ]
            list_fields = {"legal_issues", "statutes_cited", "precedents_cited"}

            for title, key in sections:
                val = judgment.get(key, "")
                if not val: continue
                val = clean_text(val)
                st.markdown(f'<div class="j-head">{title}</div>', unsafe_allow_html=True)
                if key in list_fields:
                    items = split_items(val)
                    for n, item in enumerate(items, 1):
                        st.markdown(f'<div class="j-item"><span class="j-item-num">{n}.</span> {clean_text(item)}</div>', unsafe_allow_html=True)
                elif isinstance(val, list):
                    for n, item in enumerate(val, 1):
                        st.markdown(f'<div class="j-item"><span class="j-item-num">{n}.</span> {clean_text(item)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="j-body">{val}</div>', unsafe_allow_html=True)

            st.markdown('<div class="thinline"></div>', unsafe_allow_html=True)
            c1, c2, _ = st.columns([1, 1, 3])
            with c1:
                st.download_button("↓ json", json.dumps(judgment, indent=2, ensure_ascii=False),
                    file_name=f"judgment_{uploaded.name.replace('.pdf','')}.json", mime="application/json")
            with c2:
                lines = [f"judgment — {uploaded.name}", f"{datetime.now().strftime('%Y-%m-%d %H:%M')}", f"classification: {label}", "—" * 40]
                if precedents:
                    lines.append(f"\nRELATED PRECEDENTS")
                    for p in precedents: lines.append(f"  - {p['source_file']} ({int(p['score']*100)}% match)")
                for title, key in sections:
                    v = judgment.get(key, "")
                    if isinstance(v, list): v = "\n".join(f"  {i+1}. {x}" for i, x in enumerate(v))
                    if v: lines.append(f"\n{title.upper()}\n{v}")
                st.download_button("↓ text", "\n".join(lines),
                    file_name=f"judgment_{uploaded.name.replace('.pdf','')}.txt", mime="text/plain")

elif label in ("DEAD", "STAGNANT", "REPETITIVE"):
    st.markdown('<div class="thinline"></div>', unsafe_allow_html=True)
    notes = {
        "DEAD": "this case has no practical path to resolution.",
        "STAGNANT": "this case exists but hasn't moved in a long time.",
        "REPETITIVE": "this issue was already decided in a prior case.",
    }
    st.markdown(f'<div class="reason-text">{notes.get(label, "")} judgment generation is for active cases only.</div>', unsafe_allow_html=True)

st.markdown(f'<div class="meta">{len(text):,} chars · classified in {cls_time:.1f}s{f" · judgment in {jdg_time:.1f}s" if label == "ACTIVE" and jdg_time > 0 else ""}{" · " + str(len(precedents)) + " precedents" if precedents else ""} · {backend}</div>', unsafe_allow_html=True)