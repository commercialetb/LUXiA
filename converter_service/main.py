from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response
import os, tempfile, subprocess

app = FastAPI(title="DWG2DXF Converter", version="1.0")

ODA_BIN = os.getenv("ODA_FILECONVERTER_BIN", "ODAFileConverter")
DXF_VERSION = os.getenv("ODA_DXF_VERSION", "ACAD2018")

@app.get("/health")
def health():
    return {"ok": True, "oda_bin": ODA_BIN, "dxf_version": DXF_VERSION}

@app.post("/")
async def convert_octet_stream(req: Request, x_filename: str | None = Header(default="drawing.dwg")):
    """Accepts application/octet-stream DWG bytes (compatible with engine _dwg_to_dxf)."""
    raw = await req.body()
    if not raw or len(raw) < 1024:
        raise HTTPException(status_code=400, detail="Empty/invalid DWG")
    filename = x_filename or "drawing.dwg"
    if not filename.lower().endswith(".dwg"):
        filename += ".dwg"

    with tempfile.TemporaryDirectory() as td:
        in_dir = os.path.join(td, "in")
        out_dir = os.path.join(td, "out")
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        in_path = os.path.join(in_dir, filename)
        with open(in_path, "wb") as f:
            f.write(raw)

        cmd = [ODA_BIN, in_dir, out_dir, DXF_VERSION, "DXF", "0", "1"]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ODAFileConverter failed: {p.stderr[:400]}")

        dxf_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.lower().endswith(".dxf")]
        if not dxf_files:
            raise HTTPException(status_code=500, detail="No DXF produced")
        dxf_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        with open(dxf_files[0], "rb") as f:
            dxf_bytes = f.read()

    return Response(content=dxf_bytes, media_type="application/dxf")
