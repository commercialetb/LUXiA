# LuxIA Engine (Vercel)

## Deploy gratuito su Vercel (FastAPI)

1) Repo GitHub con questa struttura:
- api/index.py
- apps/engine/engine_api.py
- requirements.txt
- vercel.json

2) Vercel:
- New Project → Importa repo
- Framework: Other
- Build Command: (vuoto)
- Output Directory: (vuoto)

3) Variabili env (solo se usi LLM/Vision):
- LLM_PROVIDER, QWEN_*/GROQ_*/OLLAMA_*
- GEMINI_API_KEY oppure OPENROUTER_API_KEY+OPENROUTER_MODEL
- (opz.) DWG2DXF_URL

4) Test:
- /health
- POST /planimetry/analyze (form-data: file + options)

Nota: DWG richiede un servizio di conversione → altrimenti usa DXF.
