const MAX_DATA_URL_LENGTH = 3 * 1024 * 1024;
const MAX_IMAGE_EDGE = 2560;

function loadImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("The selected image could not be prepared for upload."));
    image.src = dataUrl;
  });
}

/**
 * Keep image JSON requests below Vercel's 4.5 MB Function body limit. Small
 * images pass through unchanged; larger ones are resized/compressed as JPEG.
 */
export async function prepareImagePayload(dataUrl: string): Promise<string> {
  if (dataUrl.length <= MAX_DATA_URL_LENGTH) return dataUrl;

  const image = await loadImage(dataUrl);
  const longestEdge = Math.max(image.naturalWidth, image.naturalHeight);
  if (!longestEdge) throw new Error("The selected image has invalid dimensions.");

  let scale = Math.min(1, MAX_IMAGE_EDGE / longestEdge);
  const qualities = [0.88, 0.8, 0.72, 0.64, 0.56];

  for (let attempt = 0; attempt < 8; attempt += 1) {
    const width = Math.max(1, Math.round(image.naturalWidth * scale));
    const height = Math.max(1, Math.round(image.naturalHeight * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("The browser could not prepare the selected image.");

    context.fillStyle = "#fff";
    context.fillRect(0, 0, width, height);
    context.drawImage(image, 0, 0, width, height);

    for (const quality of qualities) {
      const candidate = canvas.toDataURL("image/jpeg", quality);
      if (candidate.length <= MAX_DATA_URL_LENGTH) return candidate;
    }
    scale *= 0.8;
  }

  throw new Error("This image could not be reduced below the hosting request limit.");
}
