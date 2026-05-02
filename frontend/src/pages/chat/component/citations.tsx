import IconSearch from '@/assets/chat/search.svg'
import { Button, Drawer, Input } from 'antd'
import { useMemo, useState } from 'react'
import EvidencePreview from './evidence-preview'
import { toReadableEvidenceText } from './evidence-preview-utils'
import styles from './citations.module.scss'

function CitationsItem(props: {
  item: API.Reference
  index: number
  onRead: () => void
}) {
  const { item, index, onRead } = props

  const content = useMemo(() => {
    return toReadableEvidenceText(item.content_with_weight)
  }, [item.content_with_weight])

  return (
    <div className={styles['citations__item']}>
      <div className={styles['header']}>
        <div className={styles['name']} title={item.document_name}>
          {item.document_name}
        </div>
        <div className={styles['score']}>#{index + 1}</div>
      </div>

      <div className={styles['desc']}>{content}</div>

      <div className={styles['footer']}>
        <div className={styles['footer-desc']}>
          {item.location_label ||
            (item.positions?.[0]?.[0]
              ? `Page ${item.positions[0][0]}`
              : 'Page metadata unavailable')}
        </div>
        <Button className={styles['footer-button']} onClick={onRead}>
          Open excerpt
        </Button>
      </div>
    </div>
  )
}

export default function Citations(props: { list?: API.Reference[] }) {
  const { list } = props

  const [read, setRead] = useState<API.Reference | null>(null)
  const [keyword, setKeyword] = useState('')

  const filteredList = useMemo(() => {
    const normalized = keyword.trim().toLowerCase()
    if (!normalized) return list ?? []

    return (list ?? []).filter((item) => {
      const text = toReadableEvidenceText(item.content_with_weight).toLowerCase()
      return (
        item.document_name.toLowerCase().includes(normalized) ||
        text.includes(normalized)
      )
    })
  }, [keyword, list])

  return (
    <div className={styles['citations']}>
      <div className={styles['citations__search']}>
        <Input
          value={keyword}
          onChange={(event) => setKeyword(event.target.value)}
          placeholder="Search evidence"
          suffix={<img src={IconSearch} alt="search" />}
        />
      </div>

      <div className={styles['citations__title']}>
        {`Selected evidence (${filteredList.length})`}
      </div>

      <div className={styles['citations__list']}>
        {filteredList.map((item, index) => (
          <CitationsItem
            key={`${item.document_id}-${index}`}
            item={item}
            index={index}
            onRead={() => setRead(item)}
          />
        ))}
      </div>

      <Drawer
        title={read?.document_name ?? ''}
        width={800}
        onClose={() => setRead(null)}
        open={!!read}
        destroyOnClose
        rootClassName={styles['citations__drawer']}
      >
        {read ? <EvidencePreview reference={read} /> : null}
      </Drawer>
    </div>
  )
}
