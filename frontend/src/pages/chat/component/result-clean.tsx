import IconCopy from '@/assets/chat/copy.svg'
import IconRefresh from '@/assets/chat/refresh.svg'
import IconShare from '@/assets/chat/share.svg'
import IconTip from '@/assets/chat/tip.svg'
import Markdown from '@/components/markdown'
import { ArrowRightOutlined } from '@ant-design/icons'
import { Button, Dropdown } from 'antd'
import classNames from 'classnames'
import dayjs from 'dayjs'
import { TokenizerAndRendererExtension } from 'marked'
import { useCallback, useMemo } from 'react'
import styles from './result-clean.module.scss'

function normalizeCitationMarkers(value?: string) {
  if (!value) return value

  const normalizedBracket = value.replace(
    /##((?:\[\d+\])+)\$\$/g,
    (_, group: string) => {
      const matches = [...group.matchAll(/\d+/g)].map((item) => `##${item[0]}$$`)
      return matches.join(' ')
    },
  )

  return normalizedBracket
    .replace(/##(?:reference|citation)\[(\d+)\]\$\$/g, (_, index: string) => `##${index}$$`)
    .replace(/##(\d+)##/g, (_, index: string) => `##${index}$$`)
    .replace(/##(?:reference|citation)[_ ]?(\d+)##/g, (_, index: string) => `##${index}$$`)
    .replace(/##(?:reference|citation)[_ ]?(\d+)\$\$/g, (_, index: string) => `##${index}$$`)
    .replace(/(?<![\w(])\[(\d+)\](?!\()/g, (_, index: string) => `##${index}$$`)
}

export function Result(props: {
  item: API.ChatItem
  isEnd?: boolean
  onSend?: (text: string) => void
  onRefrence?: (index: number) => void
}) {
  const { item, isEnd, onSend, onRefrence } = props
  const normalizedThink = normalizeCitationMarkers(item.think)
  const normalizedContent = normalizeCitationMarkers(item.content)

  const shareMenu = useMemo(() => {
    return [
      {
        key: 'txt',
        label: 'Export TXT',
        onClick: async () => {
          const url = `data:text/plain;charset=utf-8,${encodeURIComponent(normalizedContent ?? '')}`
          const anchor = document.createElement('a')
          anchor.href = url
          anchor.download = 'output.txt'
          anchor.click()
        },
      },
      {
        key: 'email',
        label: 'Send to Email',
      },
    ]
  }, [item.content])

  const extensions = useMemo<TokenizerAndRendererExtension[]>(
    () => [
      {
        name: 'reference',
        level: 'inline',
        start(src) {
          return src.match(/##\d+\$\$/)?.index
        },
        tokenizer(src) {
          const match = /^##(\d+?)\$\$/.exec(src)
          if (match) {
            const [raw, index] = match
            return {
              type: 'reference',
              raw,
              index: this.lexer.inlineTokens(index),
              tokens: [],
            }
          }
        },
        renderer(token) {
          const rawIndex = Number(this.parser.parseInline(token.index))
          const normalizedIndex = rawIndex > 0 ? rawIndex - 1 : rawIndex
          const label = rawIndex > 0 ? rawIndex : rawIndex + 1
          return `<span class="refrence-token" data-refrence-index="${normalizedIndex}">[${label}]</span>`
        },
      },
    ],
    [],
  )

  const handleClick = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      const target = event.target as HTMLElement
      const index = target.getAttribute('data-refrence-index')
      if (index) {
        onRefrence?.(Number(index))
      }
    },
    [onRefrence],
  )

  return (
    <div className={styles['chat-message-result']}>
      {item.think ? (
        <Markdown
          className={classNames(
            styles['chat-message-result__think'],
            styles['chat-message-result__md'],
          )}
          value={normalizedThink}
          extensions={extensions}
          onClick={handleClick}
        />
      ) : null}

      {item.content ? (
        <Markdown
          className={styles['chat-message-result__md']}
          value={normalizedContent}
          extensions={extensions}
          onClick={handleClick}
        />
      ) : null}

      {item.error ? (
        <div className={styles['chat-message-result__error']}>{item.error}</div>
      ) : null}

      {item.loading ? null : (
        <>
          <div className={styles['chat-message-result__actions']}>
            <div className={styles['date']}>
              {dayjs(item.created_at || new Date()).format('HH:mm YYYY/MM/DD')}
            </div>

            {isEnd ? null : (
              <Button
                variant="text"
                color="primary"
                shape="circle"
                size="small"
                style={{ color: 'var(--ant-color-primary)' }}
              >
                <img src={IconRefresh} />
              </Button>
            )}

            <Button
              variant="text"
              color="primary"
              shape="circle"
              size="small"
              style={{ color: 'var(--ant-color-primary)' }}
            >
              <img src={IconTip} />
            </Button>

            <Button
              variant="text"
              color="primary"
              shape="circle"
              size="small"
              style={{ color: 'var(--ant-color-primary)' }}
            >
              <img src={IconCopy} />
            </Button>

            <Dropdown menu={{ items: shareMenu }}>
              <Button
                variant="text"
                color="primary"
                shape="circle"
                size="small"
                style={{ color: 'var(--ant-color-primary)' }}
              >
                <img src={IconShare} />
              </Button>
            </Dropdown>
          </div>

          {isEnd ? (
            <div className={styles['chat-message-result__quick-reply']}>
              {item.recommended_questions?.map((question) => (
                <Button
                  className={styles['item']}
                  key={question}
                  onClick={() => onSend?.(question)}
                >
                  <span className={styles['text']}>{question}</span>
                  <ArrowRightOutlined className={styles['arrow']} />
                </Button>
              ))}
            </div>
          ) : null}
        </>
      )}
    </div>
  )
}
