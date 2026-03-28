# 🌾 Krishi — Voice & Text Agriculture Chatbot for Indian Farmers

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Model-RAG%20Ensemble-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Languages-6-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Training%20Pairs-439K-purple?style=for-the-badge" />
</p>

> **Krishi** (कृषि — Sanskrit for "Agriculture") is an AI-powered voice and text chatbot built specifically for Indian farmers. It understands questions about crops, soil, irrigation, fertilizers, pest control, and state-specific farming practices — and answers in the farmer's own language using voice or text.

---

## 🚀 Live Demo

🔗 **[Coming Soon — Render Deployment]**

---

## ✨ Features — In Detail

### 🎙️ Voice Input (Speech to Text)
- Uses the **Web Speech API** built into modern browsers — no extra app or microphone setup needed
- The farmer simply clicks the mic button and speaks naturally in their language
- The browser transcribes the speech to text and sends it to the backend
- Works on Chrome, Edge, and Safari on both desktop and mobile
- No audio is sent to our servers — transcription happens entirely in the browser

### 🔊 Voice Output (Text to Speech)
- Every text answer is also converted to **spoken audio** using **gTTS (Google Text-to-Speech)**
- The audio is streamed back as MP3 and played automatically in the browser
- Each language has its own gTTS voice so the pronunciation sounds natural
- Farmers who cannot read can still get the full answer just by listening
- TTS is capped at 3,000 characters to keep responses fast

### 🌐 Multilingual Support (6 Indian Languages)
- The chatbot can **understand and reply** in 6 languages:
  - 🇬🇧 English, 🇮🇳 Hindi, 🌿 Tamil, 🌴 Malayalam, 🌾 Telugu, 🌸 Bengali
- Language is **auto-detected** from the script of the input text — no need to manually select
- Questions are internally translated to English for retrieval, then the answer is translated back to the user's language using **MarianMT** (Helsinki-NLP translation models)
- Additionally, the LLMs are instructed via system prompt to reply natively in the target language for more fluent output

### 🧠 RAG — Retrieval Augmented Generation
- Instead of relying purely on LLM memory, Krishi **retrieves the most relevant Q&A pairs** from a knowledge base of 439,251 agriculture-specific examples before generating an answer
- This grounds the answer in real agricultural knowledge and prevents hallucination
- The top 5 most relevant context passages are injected into the LLM prompt
- Confidence scores are calculated and the best answer across 3 LLMs is selected

### 🗺️ Indian State-Specific Knowledge
- The knowledge base includes farming data specifically filtered for:
  - Tamil Nadu, Kerala, Andhra Pradesh, Karnataka, West Bengal, Punjab, Gujarat
- Answers are tagged with state context when relevant (e.g. `[Kerala]`, `[Punjab]`)
- Crops, seasons, soil types, and government schemes differ by state — Krishi accounts for this

### 💬 Conversation Memory
- Each user gets a **session ID** that persists across multiple turns
- The last 4 user+assistant exchanges are injected into every new prompt
- This allows follow-up questions like "What about in summer?" without repeating context
- Sessions automatically expire after 1 hour of inactivity to free memory

---

## 🏗️ How It Works — Architecture in Detail

```
┌─────────────────────────────────────────────────────────────┐
│                     USER (Browser)                          │
│                                                             │
│  Speaks into mic  ──►  Web Speech API  ──►  Transcribed     │
│  OR types text                              text query      │
└──────────────────────────────┬──────────────────────────────┘
                               │  HTTP POST /api/chat
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (api.py)                   │
│                                                             │
│  1. Session Management — create/retrieve session history    │
│  2. Language Detection — detect script (Hindi/Tamil/etc.)   │
│  3. Translation — MarianMT translates query → English       │
│  4. Calls chatbot_core.predict()                            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               chatbot_core.py — RAG Engine                  │
│                                                             │
│  STEP 1: EMBEDDING                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Fine-tuned all-MiniLM-L6-v2                        │   │
│  │  Converts user query → 384-dim vector               │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  STEP 2: RETRIEVAL       ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  model.pkl — 439,251 pre-embedded Q&A pairs         │   │
│  │  Cosine similarity search → Top-5 relevant passages │   │
│  │  Retrieval score calculated                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  STEP 3: GENERATION      ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  3 LLMs run in parallel (ThreadPoolExecutor)        │   │
│  │                                                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │ Zephyr-3B   │  │ Qwen2.5-3B  │  │ TinyLlama  │  │   │
│  │  │ weight: 1.2 │  │ weight: 1.15│  │ weight:0.85│  │   │
│  │  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  │                                                     │   │
│  │  Each LLM receives:                                 │   │
│  │  - System prompt (agriculture expert persona)       │   │
│  │  - Last 4 conversation turns (history)              │   │
│  │  - Top-5 retrieved context passages                 │   │
│  │  - User question                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  STEP 4: SCORING         ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Each answer is scored by:                          │   │
│  │  - Semantic similarity to retrieved context         │   │
│  │  - Model weight multiplier                          │   │
│  │  Best scoring answer is selected                    │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               Translation & TTS (api.py)                    │
│                                                             │
│  - Answer translated back to user's language (MarianMT)    │
│  - If TTS enabled: gTTS converts answer → MP3              │
│  - Response sent back to browser                           │
│  - Audio plays automatically                               │
└─────────────────────────────────────────────────────────────┘
```

### Why 3 LLMs instead of 1?
Running an **ensemble** of 3 models and picking the best answer improves reliability. If one model hallucinates or gives a vague answer, another model's answer will score higher and be selected instead. This is especially important for agriculture where wrong advice (wrong pesticide dose, wrong fertilizer) can harm a farmer's crop.

### Why RAG instead of pure LLM?
LLMs have a knowledge cutoff and can hallucinate. By retrieving from a curated knowledge base of 439K agriculture Q&A pairs, the answer is always **grounded in verified agricultural information** rather than the model's general training.

---

## 📊 Training & Datasets — In Detail

### What was trained?
The **sentence embedding model** (`all-MiniLM-L6-v2`) was fine-tuned on agriculture-specific Q&A pairs using **MultipleNegativesRankingLoss**. This makes the model much better at understanding agriculture vocabulary compared to the generic base model.

The LLMs (Zephyr, Qwen, TinyLlama) are used **as-is** from HuggingFace — only the retrieval embeddings were fine-tuned.

### Training Configuration
| Parameter | Value |
|---|---|
| Base Model | `all-MiniLM-L6-v2` |
| Loss Function | `MultipleNegativesRankingLoss` |
| Epochs | 10 |
| Batch Size | 64 |
| Device | CUDA (GPU) |
| Output Dimensions | 384 |
| Final Model Size | ~1 GB (439K embeddings stored) |

### Data Pipeline
```
Raw datasets collected         →   618,368 pairs
Remove empty/null pairs        →       -7 pairs
Remove vague answers           →  -22,539 pairs
Remove very short/long texts   →  -96,291 pairs
Deduplicate                    → -179,117 duplicates
────────────────────────────────────────────────────
Final training pairs           →   439,251 pairs
```

### Datasets Used

#### General Agriculture Datasets
| Dataset | Pairs Kept | Topics |
|---|---|---|
| `shchoi83/agriQA` | 131,798 | General agriculture Q&A |
| `SivaResearch/Agri` | 149,499 | Formatted agriculture text |
| `AI71ai/agrillm-train-146k` | 162,825 | Multi-turn agriculture conversations |
| `KisanVaani/agriculture-qa-english-only` | 18,311 | Indian farmer questions |
| `Mahesh2841/Agriculture` | 4,986 | Crop instructions |
| `argilla/farming` | 1,695 | Farming Q&A |
| `YuvrajSingh9886/Agriculture-Irrigation` | 3,556 | Irrigation systems |
| `YuvrajSingh9886/Agriculture-Soil` | 3,127 | Soil science |
| `DARJYO/crop_optimization` | 40 | Crop optimization |

#### Indian State-Specific Datasets (filtered from KisanVaani)
| State | Pairs Kept |
|---|---|
| Tamil Nadu | 71 |
| Kerala | 52 |
| Andhra Pradesh | 179 |
| Karnataka | 96 |
| Punjab | 58 |
| West Bengal | 0 (no keyword matches found) |
| Gujarat | 0 (no keyword matches found) |

> State datasets were filtered by keyword matching on crop names, region names, and local farming terms specific to each state.

---

## 🛠️ Installation & Setup — In Detail

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU recommended (runs on CPU but slowly)
- 4GB+ RAM minimum, 8GB+ recommended
- `model.pkl` file (~1GB) — download separately from HuggingFace Hub

### Step 1 — Clone the Repository
```bash
git clone https://github.com/KUNALDEBNATH/Krishi.git
cd Krishi
```

### Step 2 — Create a Virtual Environment (recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install fastapi uvicorn[standard] gtts deep-translator
pip install sentence-transformers transformers accelerate torch
pip install huggingface_hub bitsandbytes
```

> **Note:** `bitsandbytes` enables 4-bit quantization for faster LLM loading on GPU. It requires CUDA. If you're on CPU only, skip it — the code will automatically fall back to full precision.

### Step 4 — Download `model.pkl`
The trained embedding model (~1GB) is hosted on HuggingFace Hub separately (too large for GitHub).

```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='KUNALDEBNATH/krishi-model',
    filename='model.pkl',
    local_dir='.'
)
print('model.pkl downloaded!')
"
```

### Step 5 — Set Environment Variables
```bash
# Your HuggingFace token (needed to download gated models)
export HF_TOKEN="your_hf_token_here"

# On Windows PowerShell:
$env:HF_TOKEN="your_hf_token_here"
```

Get your token from: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Step 6 — Run the Server
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Step 7 — Open the App
Open your browser and go to:
```
http://localhost:8000
```

You should see the Siri-style voice UI. Click the mic button and ask a farming question!

### Troubleshooting

| Problem | Solution |
|---|---|
| `model.pkl not found` | Make sure you downloaded it to the project root folder |
| `CUDA out of memory` | Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` |
| `bitsandbytes error` | Uninstall it — CPU fallback will be used automatically |
| `gTTS connection error` | Check your internet connection — gTTS calls Google's API |
| Port 8000 already in use | Change to `--port 8001` |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Send a message, get an answer |
| `GET` | `/api/tts` | Convert text to MP3 audio |
| `GET` | `/api/languages` | List supported languages |
| `GET` | `/api/history/{session_id}` | Get conversation history |
| `DELETE` | `/api/history/{session_id}` | Clear session history |
| `GET` | `/api/status` | Server and model health check |

### Example Chat Request
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the best fertilizer for rice?",
    "language": "en",
    "session_id": "optional-custom-id",
    "tts": false
  }'
```

### Example Response
```json
{
  "session_id": "abc-123",
  "answer": "For rice, apply NPK 17-17-17 at transplanting stage...",
  "model_used": "Qwen2.5-3B",
  "score": 0.87,
  "retrieval_score": 0.91,
  "language": "en",
  "candidates": []
}
```

---

## 📁 Project Structure

```
Krishi/
├── api.py                  # FastAPI backend — all endpoints, session management
├── chatbot_core.py         # RAG engine, LLM ensemble, language handling
├── train.py                # Embedding model fine-tuning script
├── training_report.json    # Full training run statistics
├── requirements.txt        # Python dependencies
├── static/
│   └── index.html          # Siri-style voice + text UI
├── .gitignore
└── README.md
```

> `model.pkl` and `checkpoints/` are NOT in the repo — too large for GitHub (1GB+). Download `model.pkl` separately from HuggingFace Hub.

---

## 🌍 Supported Languages

| Code | Language | Voice Output | Script Auto-detect |
|---|---|---|---|
| `en` | English | ✅ | ✅ |
| `hi` | Hindi | ✅ | ✅ |
| `ta` | Tamil | ✅ | ✅ |
| `te` | Telugu | ✅ | ✅ |
| `ml` | Malayalam | ✅ | ✅ |
| `bn` | Bengali | ✅ | ✅ |

---

## 👨‍💻 Author

**Kunal Debnath**
- GitHub: [@KUNALDEBNATH](https://github.com/KUNALDEBNATH)
- HuggingFace: [huggingface.co/KUNALDEBNATH](https://huggingface.co/KUNALDEBNATH)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
