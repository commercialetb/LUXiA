LuxIA PRO – Autonomy Full v3 (stable + pro photometry)
------------------------------------------------------
Included:
- Uses the original stable planimetry pipeline from v1 (no missing helper functions)
- SCALE sanity hard-gate: if implausible scale -> verification forced FAIL
- IES + LDT support:
  * /luminaires/upload accepts .ies OR .ldt (auto-detect)
  * Lux computation uses C-plane azimuth (non-symmetric optics supported)
- UGR:
  * /luminaires/ugr/upload to load tabulated UGR values
  * Engine prefers table (bilinear interp) -> falls back to ugr_approx
- Default luminaire embedded at engine/assets/default.ies (BE_51456-4K3.ies)

DWG:
- DWG requires external conversion service configured via DWG2DXF_URL (already in v1).
  Without it, engine returns 422 with instructions.
