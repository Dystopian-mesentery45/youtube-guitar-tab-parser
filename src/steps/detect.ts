import Anthropic from "@anthropic-ai/sdk";
import sharp from "sharp";
import { log } from "../util/logger.js";

/** Bounding box in normalized 0–1000 coordinates (origin top-left). */
export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

// Number of horizontal bands the frame is divided into for detection.
// Vision models are unreliable at precise pixel boxes but reliable at picking
// labeled coarse regions, so we locate the tab by which horizontal bands it
// occupies. Sheet music/tab in lesson videos spans (nearly) the full width, so
// we only detect the vertical extent and keep full width.
const BANDS = 12;

function prompt(): string {
  return (
    `The image is split by red lines into ${BANDS} horizontal bands numbered 0 (top) to ` +
    `${BANDS - 1} (bottom), labeled on the left. List the band numbers that contain guitar ` +
    "sheet music (a standard-notation staff and/or the TAB block of horizontal string lines " +
    "with numbers). Ignore bands showing only the performer, hands, or a fretboard close-up. " +
    'Respond with ONLY JSON: {"bands":[int,...]} (empty array if none).'
  );
}

/** Resize to 1024w and overlay a labeled horizontal-band grid, as JPEG base64. */
async function bandedImage(path: string): Promise<string> {
  const base = sharp(path).resize({ width: 1024, withoutEnlargement: true });
  const meta = await base.metadata();
  const W = meta.width ?? 1024;
  const H = meta.height ?? 576;

  let g = "";
  for (let i = 0; i <= BANDS; i++) {
    const y = Math.round((i / BANDS) * H);
    g += `<line x1="0" y1="${y}" x2="${W}" y2="${y}" stroke="red" stroke-width="2"/>`;
  }
  for (let i = 0; i < BANDS; i++) {
    const y = Math.round(((i + 0.5) / BANDS) * H);
    g +=
      `<rect x="0" y="${y - 11}" width="34" height="22" fill="red"/>` +
      `<text x="4" y="${y + 6}" fill="white" font-size="18" font-family="monospace" font-weight="bold">${i}</text>`;
  }
  const svg = Buffer.from(`<svg width="${W}" height="${H}">${g}</svg>`);
  const buf = await base.composite([{ input: svg }]).jpeg({ quality: 85 }).toBuffer();
  return buf.toString("base64");
}

async function detectBands(
  client: Anthropic,
  model: string,
  framePath: string,
): Promise<number[]> {
  const data = await bandedImage(framePath);
  const resp = await client.messages.create({
    model,
    max_tokens: 100,
    messages: [
      {
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data } },
          { type: "text", text: prompt() },
        ],
      },
    ],
  });
  const text = resp.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");
  return parseBands(text);
}

function parseBands(text: string): number[] {
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return [];
  try {
    const obj = JSON.parse(match[0]);
    if (!Array.isArray(obj.bands)) return [];
    return obj.bands
      .map((n: unknown) => Number(n))
      .filter((n: number) => Number.isInteger(n) && n >= 0 && n < BANDS);
  } catch {
    return [];
  }
}

function median(nums: number[]): number {
  const sorted = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? Math.round((sorted[mid - 1] + sorted[mid]) / 2)
    : sorted[mid];
}

/** Pick ~`sample` frames spread evenly across the list. */
function pickSample(frames: string[], sample: number): string[] {
  if (frames.length <= sample) return frames;
  const step = frames.length / sample;
  const picked: string[] = [];
  for (let i = 0; i < sample; i++) {
    picked.push(frames[Math.floor(i * step + step / 2)]);
  }
  return picked;
}

/**
 * Detect the tab region by sampling frames and asking Claude which labeled
 * horizontal bands contain sheet music, then taking the median first/last band
 * (robust to an outlier frame) as the vertical extent. Width is kept full.
 * Throws if no sampled frame shows a tab.
 */
export async function detectTabBox(
  apiKey: string,
  model: string,
  frames: string[],
  sample: number,
): Promise<Box> {
  const client = new Anthropic({ apiKey });
  const picks = pickSample(frames, sample);
  log.step(`Detecting tab region from ${picks.length} sampled frames (${model})…`);

  const results = await Promise.all(picks.map((p) => detectBands(client, model, p)));
  const nonEmpty = results.filter((b) => b.length > 0);
  log.debug(
    `bands per sample: ${results.map((b) => `[${b.join(",")}]`).join(" ")}`,
  );

  if (nonEmpty.length === 0) {
    throw new Error(
      "No guitar sheet music detected in any sampled frame. Try a larger --sample, a smaller " +
        "--interval, or verify the video actually shows tabs.",
    );
  }

  // Median of each frame's first and last music band → robust vertical extent.
  const firstBand = median(nonEmpty.map((b) => Math.min(...b)));
  const lastBand = median(nonEmpty.map((b) => Math.max(...b)));

  const box: Box = {
    x0: 0,
    y0: Math.round((firstBand / BANDS) * 1000),
    x1: 1000,
    y1: Math.round(((lastBand + 1) / BANDS) * 1000),
  };

  log.success(
    `Tab region: bands ${firstBand}–${lastBand} of ${BANDS} → y ${box.y0}–${box.y1} (of 1000), full width`,
  );
  return box;
}
