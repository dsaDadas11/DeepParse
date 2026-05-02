export function normalizeWhitespace(text: string) {
  return text
    .replace(/\r\n/g, '\n')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

function tableToText(table: HTMLTableElement) {
  const rows = Array.from(table.querySelectorAll('tr'))
    .map((row) =>
      Array.from(row.querySelectorAll('th, td'))
        .map((cell) => normalizeWhitespace(cell.textContent ?? ''))
        .filter(Boolean)
        .join(' | '),
    )
    .filter(Boolean)

  return rows.join('\n')
}

export function toReadableEvidenceText(raw: string) {
  if (!raw) return ''

  const root = document.createElement('div')
  root.innerHTML = raw

  root.querySelectorAll('table').forEach((table) => {
    const replacement = document.createElement('pre')
    replacement.textContent = tableToText(table as HTMLTableElement)
    table.replaceWith(replacement)
  })

  root.querySelectorAll('br').forEach((node) => {
    node.replaceWith('\n')
  })

  root.querySelectorAll('li').forEach((node) => {
    const text = normalizeWhitespace(node.textContent ?? '')
    node.textContent = text ? `- ${text}` : ''
  })

  root
    .querySelectorAll('p, div, section, article, pre, blockquote, li, h1, h2, h3, h4, h5, h6')
    .forEach((node) => {
      node.append('\n')
    })

  return normalizeWhitespace(root.textContent ?? raw)
}
