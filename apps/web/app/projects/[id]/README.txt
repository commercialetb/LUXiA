PATCH LuxIA — Autopilot + Brief

Cosa risolve:
- Autopilot ora funziona (AreaClient mantiene la signature corretta: { project })
- Aggiunge un campo "Istruzioni progetto (brief)" per scrivere cosa deve fare LuxIA per il progetto selezionato
- Brief salvato in locale (localStorage) per progetto, senza toccare lo schema Supabase
- Bottone unico "🚀 AUTOPILOT" + toggle "PRO Radiance" on/off

Come applicare:
1) Nel tuo repo, sostituisci questo file:
   apps/web/app/projects/[id]/AreaClient.jsx
   con quello dentro questo ZIP (stesso percorso).
2) Commit + push su GitHub.
3) Vercel redeploy.

Note:
- AUTOPILOT: genera i 3 concept (Fast Engine) e poi apre automaticamente /review.
- Se PRO Radiance è ON: dopo i concept prova ad avviare /pro/radiance/jobs (se il tuo Engine lo espone).
