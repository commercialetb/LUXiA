export default function NavBar() {
  const linkStyle = {
    color: "#93c5fd",
    textDecoration: "none",
    fontSize: 14,
    padding: "6px 10px",
    borderRadius: 10,
    border: "1px solid rgba(148,163,184,0.25)",
    background: "rgba(2,6,23,0.35)",
  };

  const wrapStyle = {
    position: "sticky",
    top: 0,
    zIndex: 50,
    width: "100%",
    display: "flex",
    gap: 10,
    alignItems: "center",
    padding: "10px 14px",
    backdropFilter: "blur(10px)",
    background: "linear-gradient(180deg, rgba(2,6,23,0.85) 0%, rgba(2,6,23,0.55) 100%)",
    borderBottom: "1px solid rgba(148,163,184,0.2)",
  };

  return (
    <div style={wrapStyle}>
      <div style={{ fontWeight: 800, color: "#fff", display: "flex", alignItems: "center", gap: 8 }}>
        💡 <span>LuxIA</span>
      </div>
      <a href="/" style={linkStyle}>Home</a>
      <a href="/studio" style={linkStyle}>Studi</a>
      <a href="/studio/new" style={linkStyle}>+ Studio</a>
      <a href="/projects" style={linkStyle}>Progetti</a>
      <a href="/projects/new" style={linkStyle}>+ Progetto</a>
      <a href="/catalog" style={linkStyle}>Catalogo</a>
      <div style={{ flex: 1 }} />
      <a href="/auth/login" style={{ ...linkStyle, borderColor: "rgba(148,163,184,0.35)" }}>Login</a>
    </div>
  );
}
