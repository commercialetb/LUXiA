export const ENGINE = process.env.NEXT_PUBLIC_ENGINE_URL || "http://127.0.0.1:8787";
export const TOKEN  = process.env.NEXT_PUBLIC_LUXIA_TOKEN || "dev-local-token";

export async function engineApi(path, options = {}) {
  const res = await fetch(ENGINE + path, {
    ...options,
    headers: { "X-LuxIA-Token": TOKEN, ...(options.headers || {}) },
  });
  const text = await res.text();
  let data = null;
  try { data = JSON.parse(text); } catch { data = { raw: text }; }
  if (!res.ok) throw new Error(data?.detail || text);
  return data;
}
