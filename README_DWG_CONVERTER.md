# DWG -> DXF converter integration (LuxIA)

## Engine expects
Engine sends:
- POST `${DWG2DXF_URL}` with body=DWG bytes (application/octet-stream)
- Header `X-Filename: <original name>`

Converter responds:
- DXF bytes

## Local dev
1) Download **ODA File Converter** from Open Design Alliance (license) and place the folder in `./ODAFileConverter/`
   - Ensure binary path inside container: `/opt/oda/ODAFileConverter` (or change env `ODA_FILECONVERTER_BIN`)
2) `docker compose up --build`
3) Engine accepts `.dwg` and converts automatically.

## Render deployment
- Deploy `converter_service` as a separate Render Web Service.
- Set on engine: `DWG2DXF_URL=https://<converter-service>/`
- (Optional) set `DWG2DXF_API_KEY` if you protect your converter (engine already supports Bearer auth).
