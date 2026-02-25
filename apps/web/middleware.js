import { NextResponse } from "next/server";
import { createServerClient } from "@supabase/ssr";

export async function middleware(request) {
  const response = NextResponse.next();

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  // If env is missing (misconfigured preview), skip middleware instead of hard-crashing.
  if (!url || !anon) return response;

  const supabase = createServerClient(url, anon, {
    cookies: {
      get(name) { return request.cookies.get(name)?.value; },
      set(name, value, options) { response.cookies.set({ name, value, ...options }); },
      remove(name, options) { response.cookies.set({ name, value: "", ...options, maxAge: 0 }); },
    },
  });

  // Refresh session if needed (keeps SSR auth in sync)
  await supabase.auth.getUser();
  return response;
}

export const config = {
  // Don't run on auth pages or API routes.
  matcher: ["/((?!_next/static|_next/image|favicon.ico|auth|api).*)"],
};
