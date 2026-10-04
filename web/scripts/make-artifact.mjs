// Turns dist/index.html into an Artifact page: the Artifact host adds its own
// <!doctype>/<html>/<head>/<body>, so we keep the <title>, font links, styles and
// script, and drop the document skeleton.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
const html = readFileSync(new URL('../dist/index.html', import.meta.url), 'utf8')
const head = html.match(/<head>([\s\S]*?)<\/head>/)[1]
const body = html.match(/<body>([\s\S]*?)<\/body>/)[1]
const keep = head
  .replace(/<meta charset[^>]*>/i, '')
  .replace(/<meta name="viewport"[^>]*>/i, '')
  .trim()
const out = `${keep}\n${body.trim()}\n`
mkdirSync(new URL('../artifact/', import.meta.url), { recursive: true })
writeFileSync(new URL('../artifact/LLM Understander.html', import.meta.url), out)
console.log(`artifact/LLM Understander.html  ${(out.length / 1024).toFixed(0)} KB`)
