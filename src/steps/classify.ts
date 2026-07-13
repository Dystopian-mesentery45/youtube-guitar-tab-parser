import Anthropic from "@anthropic-ai/sdk";
import sharp from "sharp";
import { log } from "../util/logger.js";

interface Classification {
  path: string;
  isTab: boolean;
  bar: number | null;
}

const PROMPT =
  "This is a horizontal crop of guitar sheet music (a standard-notation staff above a TAB " +
  "block of string lines with numbers). At the very start of the staff a small number prints " +
  "the measure/bar number of the first bar shown in this line — read it. Also decide whether " +
  "this crop is actually readable sheet music (not a title card, intro, or the performer only). " +
  'Respond with ONLY JSON: {"isTab": boolean, "bar": integer | null}. Set bar to null if you '+
  "cannot read a bar number.";

async function encode(path: string): Promise<string> {
  const buf = await sharp(path)
    .resize({ width: 1024, withoutEnlargement: true })
    .jpeg({ quality: 85 })
    .toBuffer();
  return buf.toString("base64");
}

async function classifyOne(
  client: Anthropic,
  model: string,
  path: string,
): Promise<Classification> {
  const data = await encode(path);
  const resp = await client.messages.create({
    model,
    max_tokens: 60,
    messages: [
      {
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data } },
          { type: "text", text: PROMPT },
        ],
      },
    ],
  });
  const text = resp.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");
  return parse(path, text);
}

function parse(path: string, text: string): Classification {
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return { path, isTab: false, bar: null };
  try {
    const obj = JSON.parse(match[0]);
    const bar = Number.isInteger(obj.bar) ? (obj.bar as number) : null;
    return { path, isTab: Boolean(obj.isTab), bar };
  } catch {
    return { path, isTab: false, bar: null };
  }
}

/**
 * For each crop, ask Claude whether it's real sheet music and what bar number is
 * printed at the start of the line. Then keep exactly one crop per distinct bar
 * number (first appearance wins), dropping non-tab crops. Crops whose bar number
 * can't be read are kept (we can't dedup them). Chronological order is preserved.
 */
export async function dedupeByBar(
  apiKey: string,
  model: string,
  crops: string[],
): Promise<string[]> {
  const client = new Anthropic({ apiKey });
  log.step(`Reading bar numbers from ${crops.length} crops (${model})…`);

  const results = await Promise.all(crops.map((c) => classifyOne(client, model, c)));
  log.debug(
    "bars: " +
      results.map((r) => (r.isTab ? (r.bar ?? "?") : "x")).join(" "),
  );

  const seenBars = new Set<number>();
  const kept: string[] = [];
  let droppedNonTab = 0;
  let droppedDup = 0;

  for (const r of results) {
    if (!r.isTab) {
      droppedNonTab++;
      continue;
    }
    if (r.bar === null) {
      kept.push(r.path);
      continue;
    }
    if (seenBars.has(r.bar)) {
      droppedDup++;
      continue;
    }
    seenBars.add(r.bar);
    kept.push(r.path);
  }

  log.success(
    `Kept ${kept.length} distinct tab lines ` +
      `(dropped ${droppedDup} duplicate bars, ${droppedNonTab} non-tab crops)`,
  );
  return kept;
}
