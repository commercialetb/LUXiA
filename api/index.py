# api/index.py
# Vercel entrypoint for FastAPI (ASGI)
from apps.engine.engine_api import app

# Optional Lambda adapter (some Vercel setups accept ASGI directly, others need a handler)
try:
    from mangum import Mangum  # type: ignore
    handler = Mangum(app)
except Exception:
    handler = app
