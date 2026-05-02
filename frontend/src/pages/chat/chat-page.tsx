import * as api from '@/api'
import ComPageLayout from '@/components/page-layout'
import ComSender from '@/components/sender'
import { ChatRole, ChatType } from '@/configs'
import { deviceActions } from '@/store/device'
import { usePageTransport } from '@/utils'
import { EditOutlined } from '@ant-design/icons'
import { useMount, useRequest, useUnmount } from 'ahooks'
import { Button, Drawer } from 'antd'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import { proxy, useSnapshot } from 'valtio'
import { sessionActions } from '../../store/session'
import ChatMessage from './component/chat-message'
import Citations from './component/citations'
import Contracts from './component/contracts'
import ChatDrawer from './component/drawer'
import EvidencePreview from './component/evidence-preview'
import Source from './component/source'
import styles from './chat-page.module.scss'
import { createChatId, createChatIdText, transportToChatEnter } from './shared'

const STARTER_PROMPTS = [
  '比较《民法典》与《合同法司法解释》在违约责任认定上的差异',
  '劳动合同中“竞业限制条款”的有效要件有哪些',
  '结合相似判例，分析本案中解除合同的法律风险与举证重点',
]

async function scrollToBottom() {
  await new Promise((resolve) => setTimeout(resolve))

  const threshold = 200
  const distanceToBottom =
    document.documentElement.scrollHeight -
    document.documentElement.scrollTop -
    document.documentElement.clientHeight

  if (distanceToBottom <= threshold) {
    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: 'smooth',
    })
  }
}

function parseDocumentsPayload(raw?: string) {
  if (!raw) {
    return {
      references: [] as API.Reference[],
      answer_audit: undefined as API.AnswerAudit | undefined,
      retrieval_trace: undefined as API.RetrievalTrace | undefined,
    }
  }

  const parsed = JSON.parse(raw) as
    | API.Reference[]
    | {
        references?: API.Reference[]
        answer_audit?: API.AnswerAudit
        retrieval_trace?: API.RetrievalTrace
      }

  if (Array.isArray(parsed)) {
    return {
      references: parsed,
      answer_audit: undefined,
      retrieval_trace: undefined,
    }
  }

  return {
    references: parsed.references ?? [],
    answer_audit: parsed.answer_audit,
    retrieval_trace: parsed.retrieval_trace,
  }
}

export default function ChatPage() {
  const { id } = useParams()
  const { data: ctx } = usePageTransport(transportToChatEnter)

  const [chat] = useState(() => {
    return proxy({
      list: [] as API.ChatItem[],
    })
  })
  const { list } = useSnapshot(chat) as { list: API.ChatItem[] }
  const [documents, setDocuments] = useState<API.Document[]>([])
  const [currentChatItem, setCurrentChatItem] = useState<API.ChatItem | null>(
    null,
  )

  const history = useRequest(
    async () => {
      const { data } = await api.session.detail({
        session_id: id!,
      })
      return data
    },
    {
      manual: true,
      onSuccess(data) {
        data.forEach((item) => {
          if (item.user_question) {
            chat.list.push({
              id: createChatId(),
              role: ChatRole.User,
              type: ChatType.Text,
              created_at: item.created_at,
              content: item.user_question,
            })
          }

          if (item.model_answer) {
            const map = new Map<string, API.Document>()
            let reference: API.Reference[] = []
            let recommended_questions: string[] = []
            let answer_audit: API.AnswerAudit | undefined
            let retrieval_trace: API.RetrievalTrace | undefined

            if (item.documents) {
              try {
                const parsedPayload = parseDocumentsPayload(item.documents)
                reference = parsedPayload.references
                answer_audit = parsedPayload.answer_audit
                retrieval_trace = parsedPayload.retrieval_trace
              } catch (error) {
                console.error(error)
              }
            }

            if (item.recommended_questions) {
              try {
                recommended_questions = JSON.parse(
                  item.recommended_questions,
                ) as string[]
              } catch (error) {
                console.error(error)
              }
            }

            reference.forEach((chunk) => {
              map.set(chunk.document_id, {
                document_id: chunk.document_id,
                document_name: chunk.document_name,
                content_with_weight: chunk.content_with_weight,
              })
            })
            const nextDocuments = Array.from(map.values())

            chat.list.push({
              id: createChatId(),
              role: ChatRole.Assistant,
              type: ChatType.Document,
              created_at: item.created_at,
              content: item.model_answer,
              think: item.think,
              reference,
              documents: nextDocuments.length ? nextDocuments : undefined,
              answer_audit,
              retrieval_trace,
              recommended_questions: recommended_questions.length
                ? recommended_questions
                : undefined,
            })
          }
        })

        setTimeout(() => {
          window.scrollTo({
            top: document.documentElement.scrollHeight,
          })
        })
      },
    },
  )

  const loading = useMemo(() => {
    return list.some((item) => item.loading) || history.loading
  }, [list, history.loading])
  const loadingRef = useRef(loading)
  loadingRef.current = loading

  useEffect(() => {
    deviceActions.setChatting(loading)
  }, [loading])

  useUnmount(() => {
    deviceActions.setChatting(false)
  })

  const sendChat = useCallback(
    async (target: API.ChatItem, message: string) => {
      setCurrentChatItem(target)
      target.loading = true
      try {
        const res = await api.session.chat({
          id: id!,
          message,
        })
        sessionActions.updateKey()

        const reader = res.data.getReader()
        if (!reader) return

        await read(reader)
      } catch (error) {
        target.error = error instanceof Error ? error.message : 'Unknown error'
        throw error
      } finally {
        target.loading = false
      }

      async function read(reader: ReadableStreamDefaultReader<Uint8Array>) {
        let temp = ''
        const decoder = new TextDecoder('utf-8')
        while (true) {
          const { value, done } = await reader.read()
          if (value) {
            temp += decoder.decode(value, { stream: true })
          }

          while (true) {
            const index = temp.indexOf('\n')
            if (index === -1) break

            const slice = temp.slice(0, index)
            temp = temp.slice(index + 1)
            if (slice.startsWith('data: ')) {
              parseData(slice)
              scrollToBottom()
            }
          }

          if (done) {
            temp += decoder.decode()
            const remainder = temp.trim()
            if (remainder.startsWith('data: ')) {
              parseData(remainder)
            }
            target.loading = false
            break
          }
        }
      }

      function parseData(slice: string) {
        try {
          const str = slice
            .trim()
            .replace(/^data: /, '')
            .trim()
          if (str === '[DONE]') {
            return
          }

          const json = JSON.parse(str)
          if (json?.content) {
            if (json.thinking) {
              target.think = `${target.think || ''}${json.content || ''}`
            } else {
              target.content = `${target.content || ''}${json.content || ''}`
            }
          }

          if (json?.documents?.length) {
            target.reference = json.documents

            const map = new Map<string, API.Document>()
            json.documents.forEach((chunk: API.Reference) => {
              map.set(chunk.document_id, {
                document_id: chunk.document_id,
                document_name: chunk.document_name,
                content_with_weight: chunk.content_with_weight,
              })
            })
            const nextDocuments = Array.from(map.values())
            target.documents = nextDocuments
            setDocuments(nextDocuments)
          }

          if (json?.answer_audit) {
            target.answer_audit = json.answer_audit
          }

          if (json?.retrieval_trace) {
            target.retrieval_trace = json.retrieval_trace
          }

          if (json?.recommended_questions?.length) {
            target.recommended_questions = json.recommended_questions
          }
        } catch {
          console.debug(slice)
        }
      }
    },
    [id],
  )

  const send = useCallback(
    async (message: string) => {
      if (loadingRef.current) return
      if (!message) return

      if (chat.list.length === 0) {
        chat.list.push({
          id: createChatId(),
          role: ChatRole.User,
          type: ChatType.Text,
          created_at: new Date().toISOString(),
          content: message,
        })

        chat.list.push({
          id: createChatId(),
          role: ChatRole.Assistant,
          type: ChatType.Document,
          created_at: new Date().toISOString(),
          documents: [],
        })
      } else {
        chat.list.push({
          id: createChatId(),
          role: ChatRole.User,
          type: ChatType.Text,
          created_at: new Date().toISOString(),
          content: message,
        })

        chat.list.push({
          id: createChatId(),
          role: ChatRole.Assistant,
          type: ChatType.Document,
          created_at: new Date().toISOString(),
          content: '',
        })
        scrollToBottom()
      }

      const target = chat.list[chat.list.length - 1]
      await sendChat(target, message)
    },
    [chat, sendChat],
  )

  useMount(async () => {
    if (ctx?.data.message) {
      send(ctx.data.message)
    } else {
      history.run()
    }
  })

  useEffect(() => {
    const handleScroll = () => {
      const anchors: {
        id: string
        top: number
        item: API.ChatItem
      }[] = []

      chat.list
        .filter((item) => item.type === ChatType.Document)
        .forEach((item, index) => {
          const anchorId = createChatIdText(item.id)
          const dom = document.getElementById(anchorId)
          if (!dom) return

          const top = dom.offsetTop
          if (index === 0 || top < window.scrollY) {
            anchors.push({ id: anchorId, top, item })
          }
        })

      if (anchors.length) {
        const current = anchors.reduce((prev, curr) =>
          curr.top > prev.top ? curr : prev,
        )

        setCurrentChatItem(current.item)
      }
    }

    window.addEventListener('scroll', handleScroll)

    return () => {
      window.removeEventListener('scroll', handleScroll)
    }
  }, [chat.list])

  const title = useMemo(() => {
    return list[0]?.content ?? 'New legal consultation'
  }, [list])

  const [read, setRead] = useState<API.Reference | null>(null)

  return (
    <ComPageLayout
      sender={
        <>
          {documents.length > 0 && <Source list={documents} />}
          <ComSender
            loading={loading}
            sessionId={id}
            onSend={send}
            onContract={() => setCurrentChatItem(null)}
          />
        </>
      }
      right={
        <>
          {currentChatItem && currentChatItem.reference?.length ? (
            <ChatDrawer title="Legal Evidence">
              <Citations list={currentChatItem.reference} />
            </ChatDrawer>
          ) : (
            <ChatDrawer title="Legal Materials">
              <Contracts list={documents} />
            </ChatDrawer>
          )}
        </>
      }
    >
      <div className={styles['chat-page']}>
        <div className={styles['chat-page__header']}>
          <div className={styles['chat-page__header-title']}>{title}</div>
          <Button type="text" shape="circle">
            <EditOutlined />
          </Button>
        </div>

        {list.length === 0 ? (
          <div className={styles['chat-page__empty']}>
            <div className={styles['chat-page__eyebrow']}>DeepParse</div>
            <h1 className={styles['chat-page__hero']}>
              Ask across your uploaded legal materials
            </h1>
            <p className={styles['chat-page__subtitle']}>
              Upload contracts, statutes, judicial interpretations, or case PDFs
              to the knowledge base, then use this workspace to compare legal
              viewpoints, inspect evidence, and validate retrieval quality.
            </p>

            <div className={styles['chat-page__starters']}>
              {STARTER_PROMPTS.map((item) => (
                <button
                  className={styles['chat-page__starter']}
                  key={item}
                  onClick={() => send(item)}
                >
                  {item}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <ChatMessage
            list={list}
            onSend={send}
            onOpenCiations={setCurrentChatItem}
            onRefrence={setRead}
          />
        )}

        <Drawer
          title={read?.document_name ?? ''}
          width={800}
          onClose={() => setRead(null)}
          open={!!read}
          destroyOnClose
          rootClassName={styles['chat-page__drawer-modal']}
        >
          {read ? <EvidencePreview reference={read} /> : null}
        </Drawer>
      </div>
    </ComPageLayout>
  )
}
