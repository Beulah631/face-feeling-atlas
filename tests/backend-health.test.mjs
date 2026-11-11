import assert from "node:assert/strict";
import { test } from "node:test";
import { checkBackendHealth, normalizeBackendHealthUrl } from "../src/lib/fer/backendHealth.ts";

test("uses a normalized base URL and requires healthy plus ready", async () => {
  assert.equal(normalizeBackendHealthUrl(" http://127.0.0.1:8000/// "), "http://127.0.0.1:8000/health");
  assert.equal(normalizeBackendHealthUrl("/"), "http://localhost/health");
  let requestedUrl = "";
  const result = await checkBackendHealth("http://127.0.0.1:8000/", {
    fetcher: async (url) => {
      requestedUrl = String(url);
      return Response.json({ status: "healthy", ready: true, models: {} });
    },
  });
  assert.equal(requestedUrl, "http://127.0.0.1:8000/health");
  assert.equal(result.status, "online");
});

test("treats HTTP errors and invalid health payloads as errors", async () => {
  const httpError = await checkBackendHealth("http://localhost:8000", {
    fetcher: async () => new Response("unavailable", { status: 503 }),
  });
  assert.equal(httpError.status, "error");
  assert.match(httpError.detail, /HTTP 503/);

  const invalid = await checkBackendHealth("http://localhost:8000", {
    fetcher: async () => Response.json({ status: "healthy" }),
  });
  assert.equal(invalid.status, "error");
  assert.match(invalid.detail, /ready/);
});

test("treats an invalid URL as an error without making a request", async () => {
  let called = false;
  const result = await checkBackendHealth("not a URL", {
    fetcher: async () => {
      called = true;
      return Response.json({ status: "healthy", ready: true });
    },
  });
  assert.equal(result.status, "error");
  assert.equal(called, false);
});

test("treats network and CORS-style fetch failures as offline with guidance", async () => {
  const result = await checkBackendHealth("http://localhost:8000", {
    fetcher: async () => {
      throw new TypeError("Failed to fetch");
    },
  });
  assert.equal(result.status, "offline");
  assert.match(result.detail, /CORS/);
});

test("treats a request timeout as offline", async () => {
  const result = await checkBackendHealth("http://localhost:8000", {
    timeoutMs: 5,
    fetcher: (_url, init) =>
      new Promise((_resolve, reject) => {
        init.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
      }),
  });
  assert.equal(result.status, "offline");
  assert.match(result.detail, /timed out/);
});
