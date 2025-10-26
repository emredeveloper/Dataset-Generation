#!/usr/bin/env node
/**
 * Fetch JSON data from a remote URL or a local file.
 *
 * The script uses the built-in `fetch` implementation available in Node 18+.
 * When executed on older Node versions it falls back to the `node-fetch`
 * package (which must be installed separately).
 */
const fs = require("fs/promises");
const path = require("path");
const { URL } = require("url");

const DEFAULT_SOURCE = "https://raw.githubusercontent.com/emredeveloper/Database/main/db.json";

function isHttpUrl(candidate) {
  try {
    const parsed = new URL(candidate);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch (error) {
    return false;
  }
}

async function ensureFetch() {
  if (typeof fetch === "function") {
    return fetch;
  }

  try {
    const { default: nodeFetch } = await import("node-fetch");
    return nodeFetch;
  } catch (error) {
    throw new Error(
      "`fetch` API kullanılabilir değil. Node 18+ sürümünü kullanın veya `npm install node-fetch` komutuyla bağımlılığı yükleyin."
    );
  }
}

async function fetchJsonFromUrl(url) {
  const fetchImpl = await ensureFetch();
  const response = await fetchImpl(url);
  if (!response.ok) {
    throw new Error(`HTTP hata! Hata kodu: ${response.status}`);
  }
  return response.json();
}

async function loadJsonFromFile(filePath) {
  const absolutePath = path.resolve(filePath);
  const contents = await fs.readFile(absolutePath, "utf-8");
  return JSON.parse(contents);
}

async function main() {
  const source = process.argv[2] ?? DEFAULT_SOURCE;

  try {
    const data = isHttpUrl(source)
      ? await fetchJsonFromUrl(source)
      : await loadJsonFromFile(source);

    console.log("Alınan Veri:");
    console.log(JSON.stringify(data, null, 2));
  } catch (error) {
    console.error("Hata:", error.message);
    process.exitCode = 1;
  }
}

main();
