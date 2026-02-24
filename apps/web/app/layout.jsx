export const metadata = { title: "LuxIA SaaS MVP" };

export default function RootLayout({ children }) {
  return (
    <html lang="it">
      <body style={{ fontFamily: "ui-sans-serif, system-ui", margin: 0, background: "#0b1220", color: "#e5e7eb" }}>
        <style>{`
  .btn{
    background:#1d4ed8;
    color:white;
    border:0;
    padding:10px 12px;
    border-radius:10px;
    font-weight:800;
    cursor:pointer;
  }
  .btn:disabled{opacity:.6; cursor:not-allowed;}
  .btn:hover{filter:brightness(1.05);}
`}</style>
{children}

      </body>
    </html>
  );
}
