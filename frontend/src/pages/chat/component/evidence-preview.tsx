import { FileTextOutlined, TableOutlined } from '@ant-design/icons'
import styles from './evidence-preview.module.scss'

function normalizeWhitespace(text: string) {
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

export default function EvidencePreview(props: { reference: API.Reference }) {
  const { reference } = props

  const page = reference.positions?.[0]?.[0]
  const hasTable = /<table[\s>]/i.test(reference.content_with_weight)
  const content = toReadableEvidenceText(reference.content_with_weight)
  const metaTags = [
    reference.company,
    reference.report_period,
    reference.report_type,
    reference.source,
  ].filter(Boolean)

  return (
    <div className={styles['evidence-preview']}>
      <div className={styles['evidence-preview__meta']}>
        <div className={styles['evidence-preview__tag']}>
          <FileTextOutlined />
          {reference.document_name}
        </div>
        <div className={styles['evidence-preview__tag']}>
          {page ? `Page ${page}` : 'Page metadata unavailable'}
        </div>
        {metaTags.map((tag) => (
          <div className={styles['evidence-preview__tag']} key={tag}>
            {tag}
          </div>
        ))}
        {hasTable ? (
          <div className={styles['evidence-preview__tag']}>
            <TableOutlined />
            Table excerpt
          </div>
        ) : null}
      </div>

      <div className={styles['evidence-preview__note']}>
        This panel shows the retrieved evidence chunk, not a rendered PDF page.
        Table layout is normalized into readable text for inspection.
      </div>

      <div className={styles['evidence-preview__content']}>
        <pre className={styles['evidence-preview__text']}>{content}</pre>
      </div>
    </div>
  )
}
