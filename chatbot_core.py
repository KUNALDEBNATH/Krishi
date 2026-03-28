"""
Agriculture Chatbot Core — v10  [API-READY + HISTORY + LANGUAGE-NATIVE]
═══════════════════════════════════════════════════════════════════════
  • Language-native generation — system prompt instructs models to reply
    directly in the selected language (no post-hoc translation needed).
    Translation is kept as a fallback only if native output is detected
    to still be in English.
  • Conversation history — last N turns are injected into every prompt
    so all generators reason over the full dialogue context.
  • Module-only — no chat loop here; imported by api.py and cli.py.
═══════════════════════════════════════════════════════════════════════
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import re
import sys
import time
import inspect
import pickle
import tempfile
import threading
import warnings
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore", message=r".*`max_new_tokens`.*`max_length`.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*torch_dtype.*deprecated.*", category=UserWarning)

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import transformers
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    MarianMTModel,
    MarianTokenizer,
)

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Auth
# NOTE: prefer passing via env var HF_TOKEN rather than hardcoding.
# ─────────────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    # Fallback to hardcoded token for backward compatibility
    HF_TOKEN = "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"

try:
    from huggingface_hub import login as hf_login
    hf_login(token=HF_TOKEN, add_to_git_credential=False)
    print("HuggingFace: authenticated")
except Exception as e:
    print(f"WARNING: HuggingFace login: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Adaptive dtype kwarg
# ─────────────────────────────────────────────────────────────────────────────
def _probe_dtype_kwarg() -> str:
    try:
        sig    = inspect.signature(AutoModelForCausalLM.from_pretrained)
        params = set(sig.parameters.keys())
        if "dtype" in params and "torch_dtype" not in params:
            return "dtype"
        if "torch_dtype" in params:
            return "torch_dtype"
    except Exception:
        pass
    try:
        from packaging import version as pv
        if pv.parse(transformers.__version__) >= pv.parse("4.45.0"):
            return "dtype"
    except ImportError:
        parts = transformers.__version__.split(".")
        if (int(parts[0]), int(parts[1])) >= (4, 45):
            return "dtype"
    return "torch_dtype"

_DTYPE_KWARG_NAME: str = _probe_dtype_kwarg()
print(f"transformers {transformers.__version__} | dtype kwarg: '{_DTYPE_KWARG_NAME}='")

def _make_dtype_kwarg(dtype_value: torch.dtype) -> dict:
    return {_DTYPE_KWARG_NAME: dtype_value}

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PKL            = "model.pkl"
BASE_MODEL           = "all-MiniLM-L6-v2"
CONFIDENCE_THRESHOLD = 0.45
TOP_K                = 5
MAX_HISTORY_TURNS    = 4        # last N user+assistant pairs injected into prompts
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"

GENERATOR_MODELS = [
    {
        "id": "zephyr", "name": "stabilityai/stablelm-zephyr-3b",
        "label": "Zephyr-3B", "template": "chatml", "weight": 1.2, "extra_kwargs": {},
    },
    {
        "id": "qwen",   "name": "Qwen/Qwen2.5-3B-Instruct",
        "label": "Qwen2.5-3B","template": "chatml", "weight": 1.15,"extra_kwargs": {},
    },
    {
        "id": "tinyllama","name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "label": "TinyLlama","template": "llama",  "weight": 0.85,"extra_kwargs": {},
    },
]

LANGUAGES: Dict[str, Dict] = {
    "en": {"name": "English",   "gtts_lang": "en", "marian": None,
           "native_instruction": ""},
    "bn": {"name": "Bengali",   "gtts_lang": "bn", "marian": "Helsinki-NLP/opus-mt-en-bn",
           "native_instruction": "আপনার সম্পূর্ণ উত্তর বাংলায় লিখুন। ইংরেজি ব্যবহার করবেন না।"},
    "hi": {"name": "Hindi",     "gtts_lang": "hi", "marian": "Helsinki-NLP/opus-mt-en-hi",
           "native_instruction": "अपना पूरा उत्तर हिंदी में दें। अंग्रेज़ी का प्रयोग न करें।"},
    "ta": {"name": "Tamil",     "gtts_lang": "ta", "marian": "Helsinki-NLP/opus-mt-en-ta",
           "native_instruction": "உங்கள் முழு பதிலையும் தமிழில் எழுதுங்கள். ஆங்கிலம் பயன்படுத்தாதீர்கள்."},
    "te": {"name": "Telugu",    "gtts_lang": "te", "marian": "Helsinki-NLP/opus-mt-en-te",
           "native_instruction": "మీ పూర్తి సమాధానం తెలుగులో ఇవ్వండి. ఇంగ్లీష్ వాడకండి."},
    "ml": {"name": "Malayalam", "gtts_lang": "ml", "marian": "Helsinki-NLP/opus-mt-en-ml",
           "native_instruction": "നിങ്ങളുടെ മുഴുവൻ ഉത്തരവും മലയാളത്തിൽ എഴുതുക. ഇംഗ്ലീഷ് ഉപയോഗിക്കരുത്."},
}

# ─────────────────────────────────────────────────────────────────────────────
# System prompt builder
# ─────────────────────────────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = (
    "You are an expert agriculture assistant with deep knowledge of crops, soil, "
    "fertilizers, irrigation, pest control, livestock, and Indian state-specific "
    "farming practices (Tamil Nadu, Kerala, Andhra Pradesh, Karnataka, West Bengal, "
    "Punjab, Gujarat).\n"
    "Rules:\n"
    "1. ANALYSE the question type first, then match your format exactly:\n"
    "   * Definition/What is -> 1-2 sentences, clear and direct.\n"
    "   * Yes/No/How much   -> answer first, then 1-2 lines of reason.\n"
    "   * Steps/How to      -> numbered list, max 5 steps, each <=15 words.\n"
    "   * Pros/Cons         -> 2-3 bullet points per side, no padding.\n"
    "   * Comparison        -> side-by-side key differences only.\n"
    "   * Recommendation    -> one specific answer + one-line justification.\n"
    "2. TOKEN BUDGET: you have 150 tokens. Be sharp and complete within that.\n"
    "   Stop the moment the question is answered. Never repeat or pad.\n"
    "3. Never say 'consult a specialist' as the ONLY answer.\n"
    "4. Use the context but answer in your own words. Be specific and practical.\n"
    "5. If state-specific context is tagged (e.g. [WestBengal]), prioritize "
    "   region-appropriate advice (local varieties, climate, seasons).\n"
    "6. Consider previous conversation turns when answering follow-up questions.\n"
    "7. Never start with filler phrases like 'Great question', 'Sure!', or 'Of course'."
)

def _build_system_prompt(lang_code: str = "en") -> str:
    """Return system prompt with embedded native-language instruction."""
    base = _BASE_SYSTEM_PROMPT
    native_instr = LANGUAGES.get(lang_code, LANGUAGES["en"])["native_instruction"]
    if native_instr:
        lang_name = LANGUAGES[lang_code]["name"]
        base += (
            f"\n\nLANGUAGE REQUIREMENT (CRITICAL): "
            f"You MUST respond ENTIRELY in {lang_name}. "
            f"Do NOT use English in your response. "
            f"{native_instr}"
        )
    return base

# ─────────────────────────────────────────────────────────────────────────────
# Startup checks + Load retriever
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PKL):
    print(f"{MODEL_PKL} not found. Run train.py first.")
    sys.exit(1)

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\nLoading retriever from {MODEL_PKL}...")
with open(MODEL_PKL, "rb") as f:
    saved = pickle.load(f)

retriever = SentenceTransformer(BASE_MODEL, device=DEVICE)
retriever.load_state_dict(saved["model_state"])
retriever.eval()

question_embeddings = saved["embeddings"]
questions           = saved["questions"]
answers             = saved["answers"]
state_tags          = saved.get("state_tags", {})
print(f"Retriever loaded. Knowledge base: {len(questions):,} Q&A pairs")

# ─────────────────────────────────────────────────────────────────────────────
# Helper fixups
# ─────────────────────────────────────────────────────────────────────────────
def _fix_pad_token(tokenizer, model):
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

def _fix_generation_config(model):
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None

def _check_device_placement(model, label: str):
    if DEVICE != "cuda":
        return
    try:
        devices = {p.device.type for p in model.parameters()}
        status  = "on GPU ✓" if "cuda" in devices else "WARNING: on CPU"
        print(f"  [{label}] {status}")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Safe model loader (Strategies 1→2→3)
# ─────────────────────────────────────────────────────────────────────────────
def _safe_load_model(gm: dict, bnb_config: BitsAndBytesConfig):
    name          = gm["name"]
    label         = gm["label"]
    merged_kwargs = {"trust_remote_code": True, **gm.get("extra_kwargs", {})}

    if DEVICE == "cuda":
        try:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=HF_TOKEN)
            mdl = AutoModelForCausalLM.from_pretrained(
                name, quantization_config=bnb_config, device_map={"": 0},
                token=HF_TOKEN, **merged_kwargs)
            mdl.eval(); _fix_pad_token(tok, mdl); _fix_generation_config(mdl)
            _check_device_placement(mdl, label)
            return tok, mdl
        except Exception as e1:
            print(f"    [{label}] 4-bit failed: {e1} → trying float16...")

    if DEVICE == "cuda":
        try:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=HF_TOKEN)
            mdl = AutoModelForCausalLM.from_pretrained(
                name, **_make_dtype_kwarg(torch.float16), device_map="auto",
                token=HF_TOKEN, **merged_kwargs)
            mdl.eval(); _fix_pad_token(tok, mdl); _fix_generation_config(mdl)
            _check_device_placement(mdl, label)
            print(f"    [{label}] loaded float16")
            return tok, mdl
        except Exception as e2:
            print(f"    [{label}] float16 failed: {e2} → trying CPU...")

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=HF_TOKEN)
    mdl = AutoModelForCausalLM.from_pretrained(
        name, **_make_dtype_kwarg(torch.float32), device_map="cpu",
        token=HF_TOKEN, **merged_kwargs)
    mdl.eval(); _fix_pad_token(tok, mdl); _fix_generation_config(mdl)
    print(f"    [{label}] loaded on CPU (slow)")
    return tok, mdl

# ─────────────────────────────────────────────────────────────────────────────
# Load all generators
# ─────────────────────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
)

generators: Dict = {}
for gm in GENERATOR_MODELS:
    print(f"\nLoading [{gm['label']}]: {gm['name']} ...")
    try:
        tok, mdl = _safe_load_model(gm, bnb_config)
        generators[gm["id"]] = {"tokenizer": tok, "model": mdl, "meta": gm}
        print(f"  [{gm['label']}] ready ✓")
    except Exception as e:
        print(f"  [{gm['label']}] FAILED: {e}")

if not generators:
    print("No generators loaded. Exiting.")
    sys.exit(1)

print(f"\n{len(generators)}/{len(GENERATOR_MODELS)} generators loaded.")

# ─────────────────────────────────────────────────────────────────────────────
# Translation cache (lazy MarianMT)
# ─────────────────────────────────────────────────────────────────────────────
_translation_cache: Dict[str, Tuple] = {}

def _get_translator(lang_code: str):
    if lang_code == "en" or lang_code not in LANGUAGES:
        return None, None
    if lang_code in _translation_cache:
        return _translation_cache[lang_code]
    marian_name = LANGUAGES[lang_code]["marian"]
    if not marian_name:
        return None, None
    try:
        t = MarianTokenizer.from_pretrained(marian_name, token=HF_TOKEN)
        m = MarianMTModel.from_pretrained(marian_name, token=HF_TOKEN)
        m.eval()
        _translation_cache[lang_code] = (t, m)
        return t, m
    except Exception as e:
        print(f"  Translator load failed ({lang_code}): {e}")
        return None, None

def translate_to(text: str, lang_code: str) -> str:
    """
    Translate text to lang_code. Always returns a string.
    Engine priority:
      1. deep_translator (Google Translate) — reliable, handles long text
      2. googletrans — fallback if deep_translator missing
      3. MarianMT    — offline fallback (slower, less accurate)
    """
    if lang_code == "en" or not text.strip():
        return text

    # ── Engine 1: deep_translator (Google) ────────────────────────────────
    try:
        from deep_translator import GoogleTranslator
        MAX_CHARS = 4500
        if len(text) <= MAX_CHARS:
            result = GoogleTranslator(source="en", target=lang_code).translate(text)
            if result and result.strip():
                print(f"  [Translate→{lang_code}] deep_translator OK ({len(result)} chars)")
                return result.strip()
        else:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks, cur = [], ""
            for s in sentences:
                if len(cur) + len(s) + 1 <= MAX_CHARS:
                    cur = (cur + " " + s).strip()
                else:
                    chunks.append(cur); cur = s
            if cur:
                chunks.append(cur)
            translated_parts = []
            for chunk in chunks:
                r = GoogleTranslator(source="en", target=lang_code).translate(chunk)
                translated_parts.append(r.strip() if r else chunk)
            result = " ".join(translated_parts)
            print(f"  [Translate→{lang_code}] deep_translator chunked OK")
            return result
    except ImportError:
        print(f"  [Translate→{lang_code}] deep_translator not installed — trying googletrans")
    except Exception as e:
        print(f"  [Translate→{lang_code}] deep_translator error: {e} — trying googletrans")

    # ── Engine 2: googletrans ──────────────────────────────────────────────
    try:
        from googletrans import Translator
        result = Translator().translate(text, src="en", dest=lang_code)
        if result and result.text and result.text.strip():
            print(f"  [Translate→{lang_code}] googletrans OK")
            return result.text.strip()
    except ImportError:
        print(f"  [Translate→{lang_code}] googletrans not installed — trying MarianMT")
    except Exception as e:
        print(f"  [Translate→{lang_code}] googletrans error: {e} — trying MarianMT")

    # ── Engine 3: MarianMT (offline) ──────────────────────────────────────
    tok, mdl = _get_translator(lang_code)
    if tok is not None:
        try:
            batch = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                out = mdl.generate(**batch, num_beams=4, max_new_tokens=512)
            result = tok.decode(out[0], skip_special_tokens=True)
            if result and result.strip():
                print(f"  [Translate→{lang_code}] MarianMT OK")
                return result.strip()
        except Exception as e:
            print(f"  [Translate→{lang_code}] MarianMT error: {e}")

    print(f"  [Translate→{lang_code}] ALL engines failed — returning English")
    return text

def _is_still_english(text: str) -> bool:
    """Heuristic: if >70% chars are ASCII letters, treat as English."""
    if not text:
        return True
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    return (ascii_letters / max(len(text), 1)) > 0.70

# ─────────────────────────────────────────────────────────────────────────────
# TTS (server-side playback for CLI / speak() callers)
# ─────────────────────────────────────────────────────────────────────────────
def speak(text: str, lang_code: str = "en") -> None:
    try:
        from gtts import gTTS
        gtts_lang = LANGUAGES.get(lang_code, LANGUAGES["en"])["gtts_lang"]
        tts       = gTTS(text=text, lang=gtts_lang, slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            tmp = fp.name
        tts.save(tmp)
        def _play():
            try:
                import playsound; playsound.playsound(tmp)
            except Exception:
                if sys.platform == "linux":
                    os.system(f"mpg123 -q {tmp} 2>/dev/null || aplay {tmp} 2>/dev/null")
                elif sys.platform == "darwin":
                    os.system(f"afplay {tmp}")
                elif sys.platform == "win32":
                    os.system(f"start /min wmplayer {tmp}")
            finally:
                try: os.unlink(tmp)
                except Exception: pass
        threading.Thread(target=_play, daemon=True).start()
    except Exception as e:
        print(f"  [TTS error] {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def retrieve(user_input: str) -> Tuple[float, List[Dict]]:
    user_emb     = retriever.encode(user_input, convert_to_numpy=True, normalize_embeddings=True)
    similarities = np.dot(question_embeddings, user_emb)
    top_k_idx    = np.argsort(similarities)[::-1][:TOP_K]
    best_score   = float(similarities[int(top_k_idx[0])])
    blocks       = []
    for idx in top_k_idx:
        score = float(similarities[idx])
        if score > 0.35:
            blocks.append({
                "question": questions[idx],
                "answer":   answers[idx],
                "score":    round(score, 4),
            })
    return best_score, blocks

# ─────────────────────────────────────────────────────────────────────────────
# Conversation history formatter
# ─────────────────────────────────────────────────────────────────────────────
def _format_history(history: List[Dict], max_turns: int = MAX_HISTORY_TURNS) -> str:
    """Format last N turns for injection into prompts."""
    if not history:
        return ""
    recent = history[-(max_turns * 2):]   # keep pairs
    lines  = []
    for msg in recent:
        role  = "Farmer" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders — history-aware + language-native
# ─────────────────────────────────────────────────────────────────────────────
def _build_prompt(
    template: str,
    context: str,
    user_input: str,
    tokenizer,
    lang_code: str = "en",
    history: Optional[List[Dict]] = None,
) -> str:
    system_prompt  = _build_system_prompt(lang_code)
    history_text   = _format_history(history or [])

    user_content_parts = []
    if history_text:
        user_content_parts.append(f"Previous conversation:\n{history_text}")
    user_content_parts.append(f"Retrieved knowledge base context:\n{context}")
    user_content_parts.append(f"Current question: {user_input}")
    user_content = "\n\n".join(user_content_parts)

    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_content},
    ]

    if template == "chatml":
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    if template == "phi2":
        hist_block = f"Previous conversation:\n{history_text}\n\n" if history_text else ""
        return (
            f"Instruct: {system_prompt}\n\n"
            f"{hist_block}"
            f"Context:\n{context}\n\n"
            f"Question: {user_input}\nOutput:"
        )

    if template == "llama":
        hist_block = f"Previous conversation:\n{history_text}\n\n" if history_text else ""
        return (
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            f"{hist_block}"
            f"Retrieved context:\n{context}\n\n"
            f"Question: {user_input} [/INST]"
        )

    # Generic fallback
    hist_block = f"Previous conversation:\n{history_text}\n\n" if history_text else ""
    return (
        f"System: {system_prompt}\n\n"
        f"{hist_block}"
        f"Context:\n{context}\n\n"
        f"Question: {user_input}\nAnswer:"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Single-model generation
# ─────────────────────────────────────────────────────────────────────────────
_ARTIFACT_TOKENS = [
    "<|end|>", "<|assistant|>", "<|im_end|>", "[/INST]",
    "<<SYS>>", "<</SYS>>", "</s>", "Output:", "Instruct:",
    "<|endoftext|>", "<|system|>", "<|user|>",
]

def _generate_one(
    gen_id: str,
    context: str,
    user_input: str,
    lang_code: str = "en",
    history: Optional[List[Dict]] = None,
) -> Tuple[str, float]:
    g      = generators[gen_id]
    tok    = g["tokenizer"]
    mdl    = g["model"]
    meta   = g["meta"]
    prompt = _build_prompt(meta["template"], context, user_input, tok, lang_code, history)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    t0     = time.time()
    with torch.no_grad():
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tok.pad_token_id,
        )
    elapsed    = time.time() - t0
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw        = tok.decode(new_tokens, skip_special_tokens=True).strip()
    for art in _ARTIFACT_TOKENS:
        raw = raw.replace(art, "")
    return raw.strip(), elapsed

# ─────────────────────────────────────────────────────────────────────────────
# Parallel ensemble generation
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CandidateAnswer:
    model_id:    str
    model_label: str
    text:        str
    gen_time:    float
    score:       float      = 0.0
    breakdown:   Dict       = field(default_factory=dict)

def _generate_all_parallel(
    context: str,
    user_input: str,
    lang_code: str = "en",
    history: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[CandidateAnswer]:
    def _run(gen_id: str) -> Optional[CandidateAnswer]:
        label = generators[gen_id]["meta"]["label"]
        try:
            if verbose: print(f"    [{label}] generating...", flush=True)
            resp, elapsed = _generate_one(gen_id, context, user_input, lang_code, history)
            if verbose: print(f"    [{label}] done ({elapsed:.1f}s)  {len(resp)} chars")
            return CandidateAnswer(model_id=gen_id, model_label=label, text=resp, gen_time=elapsed)
        except Exception as e:
            if verbose: print(f"    [{label}] ERROR: {type(e).__name__}: {e}")
            return None

    candidates: List[CandidateAnswer] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(generators)) as ex:
        futures = {ex.submit(_run, gid): gid for gid in generators}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res is not None:
                candidates.append(res)
    return candidates

# ─────────────────────────────────────────────────────────────────────────────
# Chairman scorer
# ─────────────────────────────────────────────────────────────────────────────
def _extract_context_keywords(blocks: List[Dict]) -> List[str]:
    stopwords = {
        "the","a","an","is","are","was","of","in","to","and","or","it","for",
        "on","with","that","this","can","be","by","at","from","as","but","not",
        "have","has","they","you","we","do","does","which","will","also",
        "their","your","its","than","how","what","when","where","who","use",
        "using","used",
    }
    combined = " ".join(b["question"] + " " + b["answer"] for b in blocks).lower()
    tokens   = re.findall(r"[a-z]{3,}", combined)
    unigrams = [w for w in set(tokens) if len(w) >= 4 and w not in stopwords]
    bigrams  = []
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i+1]
        if w1 not in stopwords and w2 not in stopwords and len(w1)>=4 and len(w2)>=4:
            bigrams.append(f"{w1} {w2}")
    return list(dict.fromkeys(unigrams + bigrams))[:120]

def _extract_query_terms(user_input: str) -> List[str]:
    stopwords = {
        "the","a","an","is","are","was","of","in","to","and","or","it","for",
        "on","with","that","this","can","be","by","at","from","as","but","not",
        "what","how","when","where","which","who","do","does","will","use",
        "using","used","best","good","tell","me",
    }
    tokens = re.findall(r"[a-z]{3,}", user_input.lower())
    return [w for w in tokens if w not in stopwords]

def _chairman_score(
    candidate: CandidateAnswer,
    context_keywords: List[str],
    query_terms: List[str],
    model_weight: float,
) -> CandidateAnswer:
    t = candidate.text.lower()
    n = len(candidate.text)
    length_score     = 0.0 if n < 30 else (0.5 if n > 1200 else min(1.0, n/200))
    kw_hits          = sum(1 for kw in context_keywords if kw in t)
    kw_score_ctx     = min(1.0, kw_hits / max(len(context_keywords), 1))
    qt_hits          = sum(1 for qt in query_terms if qt in t)
    kw_score_qry     = min(1.0, qt_hits / max(len(query_terms), 1))
    kw_score         = 0.60 * kw_score_ctx + 0.40 * kw_score_qry
    hallucination_phrases = [
        "i don't have information","i cannot answer","i'm not sure","i do not know",
        "as an ai","as a language model","contact a specialist","consult an expert",
        "i was trained","my knowledge cutoff","no information available","i apologize but",
    ]
    penalty              = sum(1 for p in hallucination_phrases if p in t)
    hallucination_score  = max(0.0, 1.0 - 0.35 * penalty)
    specificity_markers  = [
        r"\d+\s*(kg|g|mg|l|ml|ton|ha|acre|cm|mm|cc|%|days?|week|month)",
        r"\b(variety|cultivar|hybrid|fertilizer|npk|urea|dap|potash)\b",
        r"\b(spray|apply|irrigate|harvest|sow|transplant|prune|mulch)\b",
        r"\b(soil|ph|nitrogen|phosphorus|potassium|organic matter)\b",
    ]
    spec_hits        = sum(1 for pat in specificity_markers if re.search(pat, t))
    specificity_score= min(1.0, spec_hits / 2)
    direct_score     = 1.0
    vague_starts     = ["i think","well,","hmm","so,","okay,","that's a good question","great question"]
    if any(t.startswith(v) for v in vague_starts):
        direct_score = 0.5
    weights = {"length":0.15,"keywords":0.30,"no_halluc":0.25,"specific":0.20,"direct":0.10}
    composite = (
        weights["length"]    * length_score
        + weights["keywords"]  * kw_score
        + weights["no_halluc"] * hallucination_score
        + weights["specific"]  * specificity_score
        + weights["direct"]    * direct_score
    ) * model_weight
    candidate.score     = round(composite, 4)
    candidate.breakdown = {
        "length": round(length_score,3), "kw_ctx": round(kw_score_ctx,3),
        "kw_qry": round(kw_score_qry,3), "keywords": round(kw_score,3),
        "no_halluc": round(hallucination_score,3), "specific": round(specificity_score,3),
        "direct": round(direct_score,3), "model_wt": model_weight,
        "composite": round(composite,4), "gen_time_s": round(candidate.gen_time,2),
    }
    return candidate

def chairman_select(
    candidates: List[CandidateAnswer],
    blocks: List[Dict],
    user_input: str,
) -> CandidateAnswer:
    kws         = _extract_context_keywords(blocks)
    query_terms = _extract_query_terms(user_input)
    for c in candidates:
        wt = next((gm["weight"] for gm in GENERATOR_MODELS if gm["id"] == c.model_id), 1.0)
        _chairman_score(c, kws, query_terms, wt)
    return max(candidates, key=lambda c: c.score)

# ─────────────────────────────────────────────────────────────────────────────
# Language detection (script heuristic)
# ─────────────────────────────────────────────────────────────────────────────
def detect_lang(text: str) -> Optional[str]:
    if re.search(r"[\u0980-\u09FF]", text): return "bn"
    if re.search(r"[\u0900-\u097F]", text): return "hi"
    if re.search(r"[\u0B80-\u0BFF]", text): return "ta"
    if re.search(r"[\u0C00-\u0C7F]", text): return "te"
    if re.search(r"[\u0D00-\u0D7F]", text): return "ml"
    return None

def translate_question_to_english(text: str) -> str:
    """Translate any language input to English for retrieval."""
    if not text or not text.strip():
        return text
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source="auto", target="en").translate(text)
        return result if result and result.strip() else text
    except ImportError:
        pass
    except Exception as e:
        print(f"  (deep_translator question translation failed: {e})")

    try:
        from googletrans import Translator
        tr = Translator()
        r  = tr.translate(text, dest="en")
        return r.text if r and r.text else text
    except Exception as e:
        print(f"  (googletrans question translation failed: {e})")

    return text

# ─────────────────────────────────────────────────────────────────────────────
# Master predict function
# ─────────────────────────────────────────────────────────────────────────────
def predict(
    user_input:  str,
    lang_code:   str             = "en",
    history:     Optional[List[Dict]] = None,
    tts_enabled: bool            = False,
    verbose:     bool            = True,
) -> Dict:
    """
    Main prediction function.

    Parameters
    ----------
    user_input  : question in English (translate before calling if needed)
    lang_code   : target language for the response
    history     : list of {\"role\": \"user\"|\"assistant\", \"content\": str}
    tts_enabled : whether to play TTS audio server-side
    verbose     : print chairman scorecard

    Returns
    -------
    dict with keys: answer, model_used, score, retrieval_score, candidates
    """
    user_input = user_input.strip()
    if not user_input:
        return {"answer": "Please type a question.", "model_used": None,
                "score": 0, "retrieval_score": 0, "candidates": []}

    best_score, blocks = retrieve(user_input)

    if verbose:
        print(f"\n  Retrieval — best score: {best_score:.4f} (threshold: {CONFIDENCE_THRESHOLD})")

    if best_score < CONFIDENCE_THRESHOLD:
        msg = (
            f"I don't have enough relevant information to answer confidently "
            f"(score: {best_score:.4f}). Please ask about crops, soil, "
            f"fertilizers, pests, irrigation, or livestock."
        )
        if lang_code != "en":
            translated_msg = translate_to(msg, lang_code)
            if translated_msg and not _is_still_english(translated_msg):
                msg = translated_msg
        if tts_enabled:
            speak(msg, lang_code)
        return {"answer": msg, "model_used": None, "score": 0,
                "retrieval_score": best_score, "candidates": []}

    if not blocks:
        return {"answer": "No relevant context found in knowledge base.",
                "model_used": None, "score": 0, "retrieval_score": 0, "candidates": []}

    context = "\n".join(f"- Q: {b['question']}\n  A: {b['answer']}" for b in blocks)

    if verbose:
        labels = " x ".join(g["meta"]["label"] for g in generators.values())
        print(f"\n  Ensemble (parallel): {labels}")

    candidates = _generate_all_parallel(context, user_input, lang_code, history, verbose)

    if not candidates:
        return {"answer": "All generators failed.", "model_used": None, "score": 0,
                "retrieval_score": best_score, "candidates": []}

    winner = chairman_select(candidates, blocks, user_input)

    if verbose:
        print(f"\n  Chairman Scorecard")
        for c in sorted(candidates, key=lambda x: -x.score):
            crown = " ← WINNER" if c.model_id == winner.model_id else ""
            bd    = c.breakdown
            print(
                f"    [{c.model_label:14s}]  score={c.score:.4f}{crown}  "
                f"len={bd['length']:.2f}  kw={bd['keywords']:.2f}  "
                f"halluc={bd['no_halluc']:.2f}  spec={bd['specific']:.2f}  t={bd['gen_time_s']}s"
            )
        print(f"  Selected: {winner.model_label} (score={winner.score:.4f})")

    final_answer = winner.text

    # ── GUARANTEED TRANSLATION ─────────────────────────────────────────────
    if lang_code != "en":
        lang_name = LANGUAGES[lang_code]["name"]
        if verbose:
            still_en = _is_still_english(final_answer)
            state    = "English (will translate)" if still_en else "non-English (will verify/re-translate)"
            print(f"\n  Answer appears {state} → forcing translation to {lang_name}")
        final_answer = translate_to(final_answer, lang_code)
        if verbose:
            print(f"  Final answer language check: "
                  f"{'English still!' if _is_still_english(final_answer) else lang_name + ' ✓'}")

    if tts_enabled:
        speak(final_answer, lang_code)

    return {
        "answer":          final_answer,
        "model_used":      winner.model_label,
        "score":           winner.score,
        "retrieval_score": best_score,
        "candidates":      [
            {"model": c.model_label, "score": c.score, "gen_time": c.gen_time}
            for c in sorted(candidates, key=lambda x: -x.score)
        ],
    }