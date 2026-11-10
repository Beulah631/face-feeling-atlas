export type BackendConnectionStatus = "checking" | "online" | "offline" | "error";

export interface BackendHealthPayload {
  status: string;
  ready: boolean;
  models?: Record<
    string,
    { name: string; loaded: boolean; weights_exist: boolean; load_error?: string | null }
  >;
}

export interface BackendHealthResult {
  status: Exclude<BackendConnectionStatus, "checking">;
  detail: string;
  data?: BackendHealthPayload;
}

export function normalizeBackendHealthUrl(backendUrl: string): string {
  const raw = backendUrl.trim();
  if (!raw) throw new Error("Enter a backend base URL.");

  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    throw new Error("Enter a valid absolute backend URL, such as http://127.0.0.1:8000.");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("Backend URL must use HTTP or HTTPS.");
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error("Backend URL must not contain credentials, a query string, or a fragment.");
  }

  const basePath = parsed.pathname.replace(/\/+$/, "");
  return `${parsed.origin}${basePath}/health`;
}

export async function checkBackendHealth(
  backendUrl: string,
  options: {
    fetcher?: typeof fetch;
    timeoutMs?: number;
    signal?: AbortSignal;
  } = {},
): Promise<BackendHealthResult> {
  let healthUrl: string;
  try {
    healthUrl = normalizeBackendHealthUrl(backendUrl);
  } catch (error) {
    return {
      status: "error",
      detail: error instanceof Error ? error.message : "Invalid backend URL.",
    };
  }

  const controller = new AbortController();
  let didTimeout = false;
  const timeoutMs = options.timeoutMs ?? 5000;
  const timeout = globalThis.setTimeout(() => {
    didTimeout = true;
    controller.abort();
  }, timeoutMs);
  const abortFromCaller = () => controller.abort();
  options.signal?.addEventListener("abort", abortFromCaller, { once: true });
  if (options.signal?.aborted) abortFromCaller();

  try {
    const response = await (options.fetcher ?? fetch)(healthUrl, {
      method: "GET",
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) {
      return { status: "error", detail: `Health check returned HTTP ${response.status}.` };
    }

    let payload: unknown;
    try {
      payload = await response.json();
    } catch {
      return { status: "error", detail: "Health endpoint returned invalid JSON." };
    }
    if (
      !payload ||
      typeof payload !== "object" ||
      typeof (payload as Record<string, unknown>)["status"] !== "string" ||
      typeof (payload as Record<string, unknown>)["ready"] !== "boolean"
    ) {
      return {
        status: "error",
        detail: "Health response is missing the required status and ready fields.",
      };
    }

    const data = payload as BackendHealthPayload;
    if (data.status.toLowerCase() === "healthy" && data.ready) {
      return { status: "online", detail: "Backend is healthy and ready.", data };
    }
    return {
      status: "error",
      detail: `Health endpoint responded, but reports status "${data.status}" and ready=${data.ready}.`,
      data,
    };
  } catch {
    if (didTimeout) {
      return {
        status: "offline",
        detail: `Health check timed out. The backend did not respond within ${(timeoutMs / 1000).toFixed(1)} seconds.`,
      };
    }
    if (options.signal?.aborted) {
      return {
        status: "offline",
        detail: "Health check was cancelled because the URL changed or a newer check started.",
      };
    }
    return {
      status: "offline",
      detail:
        "Could not reach /health. Check that the backend is running and that FastAPI CORS allows this frontend origin. Browser fetch does not distinguish CORS blocks from other network failures.",
    };
  } finally {
    globalThis.clearTimeout(timeout);
    options.signal?.removeEventListener("abort", abortFromCaller);
  }
}
