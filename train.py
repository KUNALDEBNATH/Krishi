from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import numpy as np
import pickle
import sys
import os
import re
import json
from datetime import datetime
from collections import defaultdict

MODEL_PKL    = "model.pkl"
REPORT_JSON  = "training_report.json"
BASE_MODEL   = "all-MiniLM-L6-v2"
EPOCHS       = 10
BATCH_SIZE   = 64
WARMUP_RATIO = 0.1
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────────────────────────────────────
# State-specific metadata tags injected into Q&A pairs so the retriever can
# rank state-relevant answers higher when the user mentions a state name.
# ─────────────────────────────────────────────────────────────────────────────
STATE_TAGS = {
    "west_bengal":   "[WestBengal]",
    "tamil_nadu":    "[TamilNadu]",
    "kerala":        "[Kerala]",
    "andhra_pradesh":"[AndhraPradesh]",
    "karnataka":     "[Karnataka]",
    "punjab":        "[Punjab]",
    "gujarat":       "[Gujarat]",
}

VAGUE_PHRASES = [
    "explained in detail", "explained about", "answered in detail",
    "given details", "suggested in details", "forwarded to the expert",
    "forwarded to expert", "contact the expert", "referred to expert",
    "consult an expert", "please contact", "not available",
    "no information", "given details about", "suggested to contact",
    "advised to contact", "please consult", "kindly contact",
    "please refer", "could not be determined", "unable to provide",
]

report = {
    "run_timestamp":         datetime.now().isoformat(),
    "device":                DEVICE,
    "base_model":            BASE_MODEL,
    "epochs":                EPOCHS,
    "batch_size":            BATCH_SIZE,
    "loss_function":         "MultipleNegativesRankingLoss",
    "datasets":              [],
    "state_datasets":        [],
    "total_removed_vague":   0,
    "total_removed_empty":   0,
    "total_length_filtered": 0,
    "deduplication":         {"before": 0, "after": 0, "removed": 0},
    "final_training_pairs":  0,
    "embeddings_shape":      None,
    "model_pkl_size_mb":     None,
    "status":                "incomplete",
}


def is_vague(text: str) -> bool:
    t = text.lower().strip()
    return any(t == p or t.startswith(p) for p in VAGUE_PHRASES)


def detect_col(columns, keys):
    col_lower = {c.lower(): c for c in columns}
    return next((col_lower[k.lower()] for k in keys if k.lower() in col_lower), None)


def make_entry(name, loader="standard"):
    return {
        "name": name, "loader": loader,
        "total_rows": 0, "kept": 0,
        "empty_skipped": 0, "vague_removed": 0, "length_removed": 0,
        "columns": [], "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOADER A — Standard flat HuggingFace datasets
# ─────────────────────────────────────────────────────────────────────────────
def load_standard(name, q_keys, a_keys, min_a=25, max_a=4000,
                  config=None, state_tag=None, report_list=None):
    qs, ans = [], []
    dr = make_entry(name)
    if report_list is None:
        report_list = report["datasets"]
    try:
        ds    = load_dataset(name, config) if config else load_dataset(name)
        split = "train" if "train" in ds else list(ds.keys())[0]
        data  = ds[split]
        dr["total_rows"] = len(data)
        dr["columns"]    = data.column_names

        q_col = detect_col(data.column_names, q_keys)
        a_col = detect_col(data.column_names, a_keys)
        tag_prefix = f"{state_tag} " if state_tag else ""
        print(f"  [{name}]  rows={len(data):,}  Q={q_col}  A={a_col}  tag={tag_prefix.strip() or 'none'}")

        if not q_col or not a_col:
            dr["error"] = f"Column mismatch — available: {data.column_names}"
            print(f"    WARNING: {dr['error']}")
            report_list.append(dr)
            return qs, ans

        for item in data:
            q = str(item.get(q_col) or "").strip()
            a = str(item.get(a_col) or "").strip()

            if not q or not a:
                dr["empty_skipped"] += 1
                report["total_removed_empty"] += 1
                continue
            if is_vague(a):
                dr["vague_removed"] += 1
                report["total_removed_vague"] += 1
                continue
            if not (min_a <= len(a) <= max_a):
                dr["length_removed"] += 1
                report["total_length_filtered"] += 1
                continue

            # Prepend state tag to question so retriever learns state relevance
            qs.append(tag_prefix + q)
            ans.append(a)

        dr["kept"] = len(qs)
        print(f"    kept={len(qs):,}  vague={dr['vague_removed']:,}  len_filter={dr['length_removed']:,}")

    except Exception as e:
        dr["error"] = str(e)
        print(f"    ERROR loading {name}: {e}")

    report_list.append(dr)
    return qs, ans


# ─────────────────────────────────────────────────────────────────────────────
# LOADER B — SivaResearch/Agri  formatted_text
# ─────────────────────────────────────────────────────────────────────────────
def load_siva_research(min_a=25, max_a=3000):
    qs, ans = [], []
    dr = make_entry("SivaResearch/Agri", loader="formatted_text")
    try:
        ds    = load_dataset("SivaResearch/Agri")
        split = "train" if "train" in ds else list(ds.keys())[0]
        data  = ds[split]
        dr["total_rows"] = len(data)
        dr["columns"]    = data.column_names
        print(f"  [SivaResearch/Agri]  rows={len(data):,}  loader=formatted_text")

        for item in data:
            raw = str(item.get("formatted_data") or "").strip()
            if not raw:
                dr["empty_skipped"] += 1
                continue

            q_match = re.search(r"questions?:\s*(.+?)(?:\nanswers?:|$)", raw, re.IGNORECASE | re.DOTALL)
            a_match = re.search(r"answers?:\s*(.+?)$", raw, re.IGNORECASE | re.DOTALL)
            q = q_match.group(1).strip() if q_match else ""
            a = a_match.group(1).strip() if a_match else ""
            q = re.sub(r"^asking (about|for|that|with)\s+", "", q, flags=re.IGNORECASE).strip()

            if not q or not a:
                dr["empty_skipped"] += 1
                report["total_removed_empty"] += 1
                continue
            if is_vague(a):
                dr["vague_removed"] += 1
                report["total_removed_vague"] += 1
                continue
            if not (min_a <= len(a) <= max_a):
                dr["length_removed"] += 1
                report["total_length_filtered"] += 1
                continue

            qs.append(q)
            ans.append(a)

        dr["kept"] = len(qs)
        print(f"    kept={len(qs):,}  vague={dr['vague_removed']:,}  len_filter={dr['length_removed']:,}")

    except Exception as e:
        dr["error"] = str(e)
        print(f"    ERROR: {e}")

    report["datasets"].append(dr)
    return qs, ans


# ─────────────────────────────────────────────────────────────────────────────
# LOADER C — AI71ai/agrillm-train-146k  multi-turn JSONL
# ─────────────────────────────────────────────────────────────────────────────
def load_agrillm(min_a=30, max_a=5000):
    qs, ans = [], []
    dr = make_entry("AI71ai/agrillm-train-146k", loader="jsonl_multiturn")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="AI71ai/agrillm-train-146k",
            filename="train.jsonl",
            repo_type="dataset",
        )
        print(f"  [AI71ai/agrillm-train-146k]  loader=jsonl_multiturn")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                dr["total_rows"] += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                turns = obj.get("turns", [])
                if not isinstance(turns, list):
                    continue
                for turn in turns:
                    if not isinstance(turn, dict):
                        continue
                    u = turn.get("user") or turn.get("human") or ""
                    a = turn.get("assistant") or turn.get("gpt") or ""
                    if isinstance(u, list): u = " ".join(str(x) for x in u)
                    if isinstance(a, list): a = " ".join(str(x) for x in a)
                    q = str(u).strip()
                    a = str(a).strip()
                    if not q or not a:
                        dr["empty_skipped"] += 1
                        continue
                    if is_vague(a):
                        dr["vague_removed"] += 1
                        continue
                    if not (min_a <= len(a) <= max_a):
                        dr["length_removed"] += 1
                        continue
                    qs.append(q)
                    ans.append(a)

        dr["kept"] = len(qs)
        print(f"    rows={dr['total_rows']:,}  kept={len(qs):,}")

    except Exception as e:
        dr["error"] = str(e)
        print(f"    ERROR: {e}")

    report["datasets"].append(dr)
    return qs, ans


# ─────────────────────────────────────────────────────────────────────────────
# LOADER D — State-specific datasets (manual/scraped/community datasets)
#
# Strategy: For each state we pull the best-available HuggingFace datasets
# that contain crops / soil / irrigation / pest questions relevant to that
# region. Where a dedicated state dataset doesn't exist we use filtered
# subsets of broader Indian agriculture datasets and tag them with the state.
#
# Each dataset is marked with a STATE_TAG so the retriever learns to
# associate regional context with state-specific answers.
# ─────────────────────────────────────────────────────────────────────────────

def load_state_datasets():
    """
    Load state-specific Indian agriculture datasets.
    Returns combined (questions, answers) lists with state tags.
    """
    sq, sa = [], []

    def add_state(qs, ans):
        sq.extend(qs)
        sa.extend(ans)

    print("\n" + "─" * 60)
    print("  Loading STATE-SPECIFIC Indian Agriculture Datasets")
    print("─" * 60)

    # ── WEST BENGAL (highest priority) ──────────────────────────────────────
    # Crops: Rice (Aman/Boro/Aus), Jute, Potato, Tea, Mustard, Vegetables
    # Focus: wetland rice, jute diseases, potato late blight, flood farming
    print("\n  [WEST BENGAL datasets]")

    # Primary: KisanVaani has many WB-specific queries (filter by keyword)
    add_state(*_load_filtered_by_keywords(
        name="KisanVaani/agriculture-qa-english-only",
        q_keys=["question"], a_keys=["answers"],
        keywords=["jute", "aman", "boro", "aus rice", "potato blight",
                  "west bengal", "bengal", "paddy flood", "mustard bengal",
                  "hooghly", "murshidabad", "jalpaiguri", "tea garden"],
        state_tag=STATE_TAGS["west_bengal"],
        report_list=report["state_datasets"],
        label="KisanVaani-WestBengal-filter",
    ))

    # Secondary: General rice/jute datasets tagged for WB
    add_state(*load_standard(
        "rajputta/agri_crop_data",
        q_keys=["question", "instruction", "input"],
        a_keys=["answer", "response", "output"],
        min_a=25, max_a=3000,
        state_tag=STATE_TAGS["west_bengal"],
        report_list=report["state_datasets"],
    ))

    add_state(*load_standard(
        "prsdm/agri_hindi_bn",          # Bengali + Hindi agri QA
        q_keys=["question", "input", "instruction"],
        a_keys=["answer", "output", "response"],
        min_a=20, max_a=3000,
        state_tag=STATE_TAGS["west_bengal"],
        report_list=report["state_datasets"],
    ))

    # ── TAMIL NADU ───────────────────────────────────────────────────────────
    # Crops: Rice, Sugarcane, Banana, Groundnut, Cotton, Coconut, Turmeric
    print("\n  [TAMIL NADU datasets]")

    add_state(*_load_filtered_by_keywords(
        name="KisanVaani/agriculture-qa-english-only",
        q_keys=["question"], a_keys=["answers"],
        keywords=["tamil", "cauvery", "rice paddy", "sugarcane", "groundnut",
                  "banana cultivation", "coconut palm", "turmeric", "cotton bollworm",
                  "drip irrigation", "kharif rabi"],
        state_tag=STATE_TAGS["tamil_nadu"],
        report_list=report["state_datasets"],
        label="KisanVaani-TamilNadu-filter",
    ))

    add_state(*load_standard(
        "ICAR-IIHR/horticultural-crops-india",
        q_keys=["question", "query", "input"],
        a_keys=["answer", "response", "output"],
        min_a=25, max_a=4000,
        state_tag=STATE_TAGS["tamil_nadu"],
        report_list=report["state_datasets"],
    ))

    # ── KERALA ───────────────────────────────────────────────────────────────
    # Crops: Rubber, Coconut, Spices (Pepper, Cardamom), Rice, Banana, Coffee
    print("\n  [KERALA datasets]")

    add_state(*_load_filtered_by_keywords(
        name="KisanVaani/agriculture-qa-english-only",
        q_keys=["question"], a_keys=["answers"],
        keywords=["rubber", "coconut", "pepper", "cardamom", "spice",
                  "kerala", "backwater", "kuttanad", "idukki", "wayanad",
                  "coffee plantation", "arecanut"],
        state_tag=STATE_TAGS["kerala"],
        report_list=report["state_datasets"],
        label="KisanVaani-Kerala-filter",
    ))

    add_state(*load_standard(
        "argilla/farming",
        q_keys=["evolved_questions", "instruction"],
        a_keys=["domain_expert_answer", "response"],
        min_a=25, max_a=10000,
        state_tag=STATE_TAGS["kerala"],
        report_list=report["state_datasets"],
    ))

    # ── ANDHRA PRADESH ───────────────────────────────────────────────────────
    # Crops: Rice, Maize, Tobacco, Chilli, Groundnut, Cotton, Aquaculture
    print("\n  [ANDHRA PRADESH datasets]")

    add_state(*_load_filtered_by_keywords(
        name="KisanVaani/agriculture-qa-english-only",
        q_keys=["question"], a_keys=["answers"],
        keywords=["andhra", "telangana", "chilli", "tobacco", "guntur",
                  "aquaculture shrimp", "krishna river", "godavari",
                  "red soil", "black cotton soil", "maize hybrid"],
        state_tag=STATE_TAGS["andhra_pradesh"],
        report_list=report["state_datasets"],
        label="KisanVaani-AndhraPradesh-filter",
    ))

    add_state(*load_standard(
        "Mahesh2841/Agriculture",
        q_keys=["instruction", "input"], a_keys=["response"],
        min_a=25, max_a=2000,
        state_tag=STATE_TAGS["andhra_pradesh"],
        report_list=report["state_datasets"],
    ))

    # ── KARNATAKA ────────────────────────────────────────────────────────────
    # Crops: Ragi, Maize, Sunflower, Sorghum, Coffee, Silk, Grapes, Pomegranate
    print("\n  [KARNATAKA datasets]")

    add_state(*_load_filtered_by_keywords(
        name="KisanVaani/agriculture-qa-english-only",
        q_keys=["question"], a_keys=["answers"],
        keywords=["karnataka", "ragi", "finger millet", "silk", "sericulture",
                  "coffee arabica", "sunflower", "pomegranate", "grape",
                  "kaveri", "deccan plateau", "bellary", "tumkur"],
        state_tag=STATE_TAGS["karnataka"],
        report_list=report["state_datasets"],
        label="KisanVaani-Karnataka-filter",
    ))

    add_state(*load_standard(
        "DARJYO/sawotiQ29_crop_optimization",
        q_keys=["instruction", "input", "question"],
        a_keys=["output", "answer", "response"],
        min_a=25, max_a=5000,
        state_tag=STATE_TAGS["karnataka"],
        report_list=report["state_datasets"],
    ))

    # ── PUNJAB ───────────────────────────────────────────────────────────────
    # Crops: Wheat, Rice, Maize, Sugarcane, Cotton, Potato
    # Famous for Green Revolution, heavy fertilizer/pesticide use
    print("\n  [PUNJAB datasets]")

    add_state(*_load_filtered_by_keywords(
        name="KisanVaani/agriculture-qa-english-only",
        q_keys=["question"], a_keys=["answers"],
        keywords=["wheat", "punjab", "haryana", "green revolution",
                  "paddy burning", "stubble", "amritsar", "ludhiana",
                  "basmati", "cotton bollworm punjab", "mandi"],
        state_tag=STATE_TAGS["punjab"],
        report_list=report["state_datasets"],
        label="KisanVaani-Punjab-filter",
    ))

    add_state(*load_standard(
        "shchoi83/agriQA",
        q_keys=["questions", "question"],
        a_keys=["answers", "answer"],
        min_a=40, max_a=3000,
        state_tag=STATE_TAGS["punjab"],
        report_list=report["state_datasets"],
    ))

    # ── GUJARAT ──────────────────────────────────────────────────────────────
    # Crops: Cotton (BT), Groundnut, Castor, Tobacco, Wheat, Cumin
    print("\n  [GUJARAT datasets]")

    add_state(*_load_filtered_by_keywords(
        name="KisanVaani/agriculture-qa-english-only",
        q_keys=["question"], a_keys=["answers"],
        keywords=["gujarat", "bt cotton", "castor", "groundnut gujarat",
                  "cumin", "saurashtra", "kutch", "anand", "drip gujarat",
                  "check dam", "rainwater harvesting gujarat"],
        state_tag=STATE_TAGS["gujarat"],
        report_list=report["state_datasets"],
        label="KisanVaani-Gujarat-filter",
    ))

    add_state(*load_standard(
        "YuvrajSingh9886/Agriculture-Irrigation-QA-Pairs-Dataset",
        q_keys=["QUESTION.question", "question", "Question", "QUESTION", "input"],
        a_keys=["ANSWER", "answer", "Answer", "output"],
        min_a=25, max_a=3000,
        state_tag=STATE_TAGS["gujarat"],
        report_list=report["state_datasets"],
    ))

    print(f"\n  State datasets total: {len(sq):,} pairs loaded")
    return sq, sa


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Filter an existing dataset by keywords (for regional specialization)
# ─────────────────────────────────────────────────────────────────────────────
def _load_filtered_by_keywords(name, q_keys, a_keys, keywords,
                                state_tag=None, report_list=None,
                                label=None, min_a=25, max_a=3000):
    """
    Load a HuggingFace dataset and keep only rows where the question
    contains at least one of the provided keywords (case-insensitive).
    Injects a state_tag prefix into the question text.
    """
    qs, ans = [], []
    entry_name = label or f"{name}_filtered"
    dr = make_entry(entry_name, loader="keyword_filter")
    if report_list is None:
        report_list = report["state_datasets"]

    tag_prefix = f"{state_tag} " if state_tag else ""
    kw_lower   = [k.lower() for k in keywords]

    try:
        ds    = load_dataset(name)
        split = "train" if "train" in ds else list(ds.keys())[0]
        data  = ds[split]
        dr["total_rows"] = len(data)
        dr["columns"]    = data.column_names

        q_col = detect_col(data.column_names, q_keys)
        a_col = detect_col(data.column_names, a_keys)

        if not q_col or not a_col:
            dr["error"] = f"Column mismatch — available: {data.column_names}"
            report_list.append(dr)
            return qs, ans

        print(f"  [{entry_name}]  rows={len(data):,}  filtering by {len(keywords)} keywords")

        for item in data:
            q = str(item.get(q_col) or "").strip()
            a = str(item.get(a_col) or "").strip()

            if not q or not a:
                dr["empty_skipped"] += 1
                continue

            q_lower = q.lower()
            if not any(kw in q_lower for kw in kw_lower):
                continue  # Not relevant to this state

            if is_vague(a):
                dr["vague_removed"] += 1
                continue
            if not (min_a <= len(a) <= max_a):
                dr["length_removed"] += 1
                continue

            qs.append(tag_prefix + q)
            ans.append(a)

        dr["kept"] = len(qs)
        print(f"    kept={len(qs):,} (keyword-matched, tagged {tag_prefix.strip()})")

    except Exception as e:
        dr["error"] = str(e)
        print(f"    ERROR in keyword filter for {name}: {e}")

    report_list.append(dr)
    return qs, ans


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Agriculture Chatbot  —  GPU Training Pipeline v2")
print("  (with State-Specific Indian Agriculture Datasets)")
print("=" * 60)
print(f"  Device     : {DEVICE}")
print(f"  Base model : {BASE_MODEL}")
print(f"  Epochs     : {EPOCHS}  |  Batch: {BATCH_SIZE}")
print(f"  Loss       : MultipleNegativesRankingLoss")
print(f"  States     : TN, KL, AP, KA, WB, PB, GJ")
print("=" * 60)
print("\n[Step 1/5] Loading datasets...\n")

all_q, all_a = [], []


def add(qs, ans):
    all_q.extend(qs)
    all_a.extend(ans)
    print(f"  >> Running total: {len(all_q):,} pairs\n")


# ── GROUP 1: Core global agriculture datasets ─────────────────────────────
print("\n  ── GLOBAL AGRICULTURE DATASETS ──")
add(*load_standard("KisanVaani/agriculture-qa-english-only",
    q_keys=["question"], a_keys=["answers"], min_a=25, max_a=2000))

add(*load_standard("Mahesh2841/Agriculture",
    q_keys=["instruction", "input"], a_keys=["response"], min_a=25, max_a=2000))

add(*load_standard("argilla/farming",
    q_keys=["evolved_questions", "instruction"],
    a_keys=["domain_expert_answer", "response"],
    min_a=25, max_a=10000))

add(*load_standard("DARJYO/sawotiQ29_crop_optimization",
    q_keys=["instruction", "input", "question"],
    a_keys=["output", "answer", "response"],
    min_a=25, max_a=5000))

add(*load_standard("YuvrajSingh9886/Agriculture-Irrigation-QA-Pairs-Dataset",
    q_keys=["QUESTION.question", "question", "Question", "QUESTION", "input"],
    a_keys=["ANSWER", "answer", "Answer", "output"],
    min_a=25, max_a=3000))

add(*load_standard("YuvrajSingh9886/Agriculture-Soil-QA-Pairs-Dataset",
    q_keys=["QUESTION.question", "question", "Question", "QUESTION"],
    a_keys=["ANSWER", "answer", "Answer"],
    min_a=25, max_a=3000))

add(*load_standard("shchoi83/agriQA",
    q_keys=["questions", "question"],
    a_keys=["answers", "answer"],
    min_a=40, max_a=3000))

# ── GROUP 2: Large corpus loaders ─────────────────────────────────────────
add(*load_siva_research(min_a=25, max_a=3000))
add(*load_agrillm(min_a=30, max_a=5000))

# ── GROUP 3: State-specific Indian Agriculture Datasets ───────────────────
print("\n  ── STATE-SPECIFIC INDIAN DATASETS ──")
state_q, state_a = load_state_datasets()
add(state_q, state_a)

print(f"\n  Total vague removed  : {report['total_removed_vague']:,}")
print(f"  Total empty skipped  : {report['total_removed_empty']:,}")
print(f"  Total length removed : {report['total_length_filtered']:,}")

print("\n[Step 2/5] Deduplicating...")
report["deduplication"]["before"] = len(all_q)

seen = set()
questions, answers = [], []
for q, a in zip(all_q, all_a):
    key = q.lower().strip()
    if key not in seen:
        seen.add(key)
        questions.append(q)
        answers.append(a)

report["deduplication"]["after"]   = len(questions)
report["deduplication"]["removed"] = len(all_q) - len(questions)
print(f"  Before : {report['deduplication']['before']:,}")
print(f"  After  : {report['deduplication']['after']:,}")
print(f"  Removed: {report['deduplication']['removed']:,}")

if not questions:
    print("\nNo valid Q&A pairs found. Exiting.")
    sys.exit(1)

report["final_training_pairs"] = len(questions)
print(f"\n  Final training pairs : {len(questions):,}")

print(f"\n[Step 3/5] Fine-tuning on {DEVICE.upper()}...")
train_examples   = [InputExample(texts=[q, a]) for q, a in zip(questions, answers)]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
model            = SentenceTransformer(BASE_MODEL, device=DEVICE)
loss             = losses.MultipleNegativesRankingLoss(model)
warmup_steps     = int(len(train_dataloader) * EPOCHS * WARMUP_RATIO)

print(f"  Training pairs  : {len(train_examples):,}")
print(f"  Steps per epoch : {len(train_dataloader):,}")
print(f"  Total steps     : {len(train_dataloader) * EPOCHS:,}")
print(f"  Warmup steps    : {warmup_steps:,}")

model.fit(
    train_objectives=[(train_dataloader, loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    show_progress_bar=True,
    checkpoint_path="checkpoints",
    checkpoint_save_steps=len(train_dataloader) * 5,
    use_amp=True,
)
print("  Fine-tuning complete.")

print(f"\n[Step 4/5] Encoding {len(questions):,} questions on GPU...")
question_embeddings = model.encode(
    questions,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
    batch_size=512,
    device=DEVICE,
)
report["embeddings_shape"] = list(question_embeddings.shape)
print(f"  Embeddings shape: {question_embeddings.shape}")

print(f"\n[Step 5/5] Saving to {MODEL_PKL}...")
with open(MODEL_PKL, "wb") as f:
    pickle.dump(
        {
            "model_state":  model.state_dict(),
            "embeddings":   question_embeddings,
            "questions":    questions,
            "answers":      answers,
            "state_tags":   STATE_TAGS,      # persisted so test.py can use them
        },
        f,
    )

size_mb                     = os.path.getsize(MODEL_PKL) / 1e6
report["model_pkl_size_mb"] = round(size_mb, 2)
report["status"]            = "complete"
print(f"  Saved -> {MODEL_PKL} ({size_mb:.1f} MB)")

with open(REPORT_JSON, "w") as f:
    json.dump(report, f, indent=2)
print(f"  Report -> {REPORT_JSON}")

print("\n" + "=" * 60)
print("  Training Summary")
print("=" * 60)
ok   = [d for d in report["datasets"] if not d.get("error")]
fail = [d for d in report["datasets"] if d.get("error")]
s_ok   = [d for d in report["state_datasets"] if not d.get("error")]
s_fail = [d for d in report["state_datasets"] if d.get("error")]

print(f"  Global datasets OK     : {len(ok)}")
print(f"  Global datasets FAILED : {len(fail)}")
print(f"  State datasets OK      : {len(s_ok)}")
print(f"  State datasets FAILED  : {len(s_fail)}")
print()
for d in ok + s_ok:
    print(f"  [OK]     {d['name']}: {d['kept']:,} pairs")
for d in fail + s_fail:
    print(f"  [FAILED] {d['name']}: {d['error']}")
print()
print(f"  Total vague removed   : {report['total_removed_vague']:,}")
print(f"  Total empty skipped   : {report['total_removed_empty']:,}")
print(f"  Total length filtered : {report['total_length_filtered']:,}")
print(f"  Duplicates removed    : {report['deduplication']['removed']:,}")
print(f"  Final training pairs  : {report['final_training_pairs']:,}")
print(f"  Embeddings shape      : {report['embeddings_shape']}")
print(f"  Model size            : {report['model_pkl_size_mb']} MB")
print(f"  Report saved to       : {REPORT_JSON}")
print("=" * 60)