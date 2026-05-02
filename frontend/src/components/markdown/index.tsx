import classNames from 'classnames'
import { Marked, Renderer, TokenizerAndRendererExtension } from 'marked'
import { useMemo } from 'react'
import './index.scss'

const SAFE_TAGS = new Set([
  'A',
  'BLOCKQUOTE',
  'BR',
  'CODE',
  'EM',
  'H1',
  'H2',
  'H3',
  'H4',
  'H5',
  'H6',
  'HR',
  'LI',
  'OL',
  'P',
  'PRE',
  'SPAN',
  'STRONG',
  'UL',
])

const SAFE_ATTRS = new Set(['class', 'data-refrence-index', 'href', 'rel', 'target'])

function isSafeHref(value: string) {
  return /^(https?:|mailto:|#)/i.test(value)
}

function sanitizeHtml(html: string) {
  const template = document.createElement('template')
  template.innerHTML = html

  const elements = Array.from(template.content.querySelectorAll('*'))
  elements.forEach((element) => {
    if (!SAFE_TAGS.has(element.tagName)) {
      element.replaceWith(document.createTextNode(element.textContent ?? ''))
      return
    }

    Array.from(element.attributes).forEach((attribute) => {
      if (!SAFE_ATTRS.has(attribute.name)) {
        element.removeAttribute(attribute.name)
        return
      }

      if (attribute.name === 'href' && !isSafeHref(attribute.value)) {
        element.removeAttribute(attribute.name)
        return
      }

      if (attribute.name === 'href') {
        element.setAttribute('target', '_blank')
        element.setAttribute('rel', 'noopener noreferrer')
      }
    })
  })

  return template.innerHTML
}

export default function Markdown(props: {
  className?: string
  value?: string
  extensions?: TokenizerAndRendererExtension[]
  onClick?: React.MouseEventHandler<HTMLDivElement>
}) {
  const { value, extensions, className, ...otherProps } = props

  const html = useMemo(() => {
    const renderer = new Renderer()

    const marked = new Marked({
      extensions,
    })
    const html = marked.parse(value ?? '', {
      gfm: false,
      renderer,
    }) as string

    return sanitizeHtml(html)
  }, [extensions, value])

  return (
    <div
      className={classNames('com-markdown', className)}
      {...otherProps}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
