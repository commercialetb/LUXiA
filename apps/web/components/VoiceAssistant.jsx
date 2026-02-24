"use client";
import { useEffect, useMemo, useRef, useState } from "react";

/**
 * LuxIA Voice Assistant (free, browser-native)
 * - TTS: window.speechSynthesis
 * - STT (dictation): Web Speech API (SpeechRecognition) where supported (Chrome/Edge)
 *
 * Props:
 *  - text: string to speak (optional)
 *  - onTranscript: (text)=>void  called when dictation returns final transcript (optional)
 *  - lang: BCP-47 language tag (default "it-IT")
 *  - label: small title (optional)
 */
export default function VoiceAssistant({
  text = "",
  onTranscript,
  lang = "it-IT",
  label = "Voce LuxIA",
}) {
  const [supportedTTS, setSupportedTTS] = useState(false);
  const [supportedSTT, setSupportedSTT] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [listening, setListening] = useState(false);
  const [err, setErr] = useState("");
  const [rate, setRate] = useState(1.0);
  const [pitch, setPitch] = useState(1.0);
  const [volume, setVolume] = useState(1.0);
  const [voiceName, setVoiceName] = useState("");

  const recogRef = useRef(null);

  useEffect(() => {
    setSupportedTTS(typeof window !== "undefined" && "speechSynthesis" in window);

    const SR = typeof window !== "undefined"
      ? (window.SpeechRecognition || window.webkitSpeechRecognition)
      : null;
    setSupportedSTT(!!SR);

    if (SR) {
      const r = new SR();
      r.lang = lang;
      r.interimResults = true;
      r.continuous = false;
      r.maxAlternatives = 1;

      r.onresult = (ev) => {
        let finalText = "";
        let interim = "";
        for (let i = ev.resultIndex; i < ev.results.length; i++) {
          const t = ev.results[i][0].transcript;
          if (ev.results[i].isFinal) finalText += t;
          else interim += t;
        }
        // Show interim as error line (lightweight) to avoid adding UI state
        if (interim) setErr("🎙️ " + interim);
        if (finalText) {
          setErr("");
          onTranscript?.(finalText.trim());
        }
      };
      r.onerror = (e) => {
        setErr("Errore voce: " + (e?.error || "unknown"));
        setListening(false);
      };
      r.onend = () => setListening(false);

      recogRef.current = r;
    }
  }, [lang]);

  const voices = useMemo(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return [];
    return window.speechSynthesis.getVoices?.() || [];
  }, [supportedTTS]);

  // Some browsers load voices async; refresh list
  useEffect(() => {
    if (!supportedTTS) return;
    const onVoices = () => { try { window.speechSynthesis.getVoices(); } catch {} };
    window.speechSynthesis.onvoiceschanged = onVoices;
    onVoices();
    return () => { window.speechSynthesis.onvoiceschanged = null; };
  }, [supportedTTS]);

  function stopSpeak() {
    try {
      window.speechSynthesis.cancel();
    } catch {}
    setSpeaking(false);
  }

  function speak(t) {
    setErr("");
    if (!supportedTTS) return setErr("TTS non supportato in questo browser.");
    const msg = (t || text || "").trim();
    if (!msg) return setErr("Nessun testo da leggere.");

    try {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(msg);
      u.lang = lang;
      u.rate = rate;
      u.pitch = pitch;
      u.volume = volume;

      // choose voice
      const v = (window.speechSynthesis.getVoices() || []).find(vv => vv.name === voiceName)
        || (window.speechSynthesis.getVoices() || []).find(vv => (vv.lang || "").toLowerCase().startsWith(lang.toLowerCase().slice(0,2)))
        || null;
      if (v) u.voice = v;

      u.onstart = () => setSpeaking(true);
      u.onend = () => setSpeaking(false);
      u.onerror = () => { setSpeaking(false); setErr("Errore TTS."); };

      window.speechSynthesis.speak(u);
    } catch (e) {
      setErr("Errore TTS: " + String(e?.message || e));
    }
  }

  function toggleListen() {
    setErr("");
    if (!supportedSTT) return setErr("Dettatura non supportata (usa Chrome/Edge).");
    if (!recogRef.current) return setErr("SpeechRecognition non inizializzato.");
    if (listening) {
      try { recogRef.current.stop(); } catch {}
      setListening(false);
      return;
    }
    try {
      setListening(true);
      recogRef.current.lang = lang;
      recogRef.current.start();
    } catch (e) {
      setListening(false);
      setErr("Impossibile avviare microfono: " + String(e?.message || e));
    }
  }

  return (
    <div style={{ padding: 12, border: "1px solid #24304a", borderRadius: 14, background: "#0b1530" }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", gap:10, flexWrap:"wrap" }}>
        <div>
          <div style={{ fontWeight: 900 }}>{label}</div>
          <div style={{ fontSize: 12, opacity: 0.8 }}>
            {supportedTTS ? "🔊 TTS ok" : "🔇 TTS non disponibile"}{" • "}
            {supportedSTT ? "🎙️ Dettatura ok" : "🎙️ Dettatura non disponibile"}
          </div>
        </div>

        <div style={{ display:"flex", gap:8, flexWrap:"wrap", alignItems:"center" }}>
          <button onClick={() => speak(text)} style={btn(speaking ? "#ef4444" : "#2563eb")}>
            {speaking ? "Stop" : "Leggi"}
          </button>
          <button onClick={toggleListen} style={btn(listening ? "#16a34a" : "#0f1a33")}>
            {listening ? "Ascolto…" : "Detta"}
          </button>
        </div>
      </div>

      <div style={{ marginTop: 10, display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:10 }}>
        <div>
          <div style={lab()}>Velocità</div>
          <input type="range" min="0.7" max="1.2" step="0.05" value={rate} onChange={(e)=>setRate(Number(e.target.value))} style={{ width:"100%" }} />
          <div style={mini()}>{rate.toFixed(2)}</div>
        </div>
        <div>
          <div style={lab()}>Tono</div>
          <input type="range" min="0.7" max="1.3" step="0.05" value={pitch} onChange={(e)=>setPitch(Number(e.target.value))} style={{ width:"100%" }} />
          <div style={mini()}>{pitch.toFixed(2)}</div>
        </div>
        <div>
          <div style={lab()}>Volume</div>
          <input type="range" min="0" max="1" step="0.05" value={volume} onChange={(e)=>setVolume(Number(e.target.value))} style={{ width:"100%" }} />
          <div style={mini()}>{volume.toFixed(2)}</div>
        </div>
      </div>

      <div style={{ marginTop: 10 }}>
        <div style={lab()}>Voce (se disponibile)</div>
        <select value={voiceName} onChange={(e)=>setVoiceName(e.target.value)} style={inp()}>
          <option value="">Auto</option>
          {(typeof window !== "undefined" ? (window.speechSynthesis?.getVoices?.() || []) : []).map(v => (
            <option key={v.name} value={v.name}>{v.name} — {v.lang}</option>
          ))}
        </select>
      </div>

      {err && <div style={{ marginTop: 10, color: "#fde68a", fontSize: 12 }}>{err}</div>}
    </div>
  );
}

const btn = (bg) => ({
  background: bg,
  border: "none",
  padding: "10px 14px",
  borderRadius: 10,
  color: "white",
  fontWeight: 900,
  cursor: "pointer",
});
const inp = () => ({
  width: "100%",
  padding: 10,
  borderRadius: 10,
  border: "1px solid #24304a",
  background: "#020617",
  color: "#e5e7eb",
});
const lab = () => ({ fontSize: 12, opacity: 0.8, marginBottom: 4 });
const mini = () => ({ fontSize: 12, opacity: 0.75, marginTop: 4 });
