import assert from "node:assert/strict";
import { test } from "node:test";
import { prepareImagePayload } from "../src/lib/fer/imagePayload.ts";

test("keeps already-bounded image payloads unchanged", async () => {
  const dataUrl = "data:image/jpeg;base64,small";
  assert.equal(await prepareImagePayload(dataUrl), dataUrl);
});

test("re-encodes oversized image payloads into a bounded JPEG", async () => {
  const originalImage = globalThis.Image;
  const originalDocument = globalThis.document;
  globalThis.Image = class {
    naturalWidth = 640;
    naturalHeight = 480;
    set src(_value) {
      this.onload();
    }
  };
  globalThis.document = {
    createElement: () => ({
      width: 0,
      height: 0,
      getContext: () => ({ fillStyle: "", fillRect() {}, drawImage() {} }),
      toDataURL: (type) => `data:${type};base64,compressed`,
    }),
  };

  try {
    const oversized = `data:image/png;base64,${"x".repeat(3 * 1024 * 1024)}`;
    const result = await prepareImagePayload(oversized);
    assert.ok(result.startsWith("data:image/jpeg;base64,"));
    assert.ok(result.length < 3 * 1024 * 1024);
  } finally {
    if (originalImage === undefined) delete globalThis.Image;
    else globalThis.Image = originalImage;
    if (originalDocument === undefined) delete globalThis.document;
    else globalThis.document = originalDocument;
  }
});
