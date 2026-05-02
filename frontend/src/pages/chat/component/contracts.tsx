import { CheckCircleFilled } from '@ant-design/icons'
import IconSearch from '@/assets/chat/search.svg'
import { Input } from 'antd'
import { useMemo, useState } from 'react'
import styles from './contracts.module.scss'

function ContractItem(props: { item: API.Document }) {
  const { item } = props

  return (
    <div className={styles['contracts__item']}>
      <div className={styles['name']} title={item.document_name}>
        {item.document_name}
      </div>
      <div className={styles['status']}>
        <CheckCircleFilled />
        Indexed
      </div>
    </div>
  )
}

export default function Contracts(props: { list: API.Document[] }) {
  const { list } = props
  const [keyword, setKeyword] = useState('')

  const filteredList = useMemo(() => {
    const normalized = keyword.trim().toLowerCase()
    if (!normalized) return list
    return list.filter((item) =>
      item.document_name.toLowerCase().includes(normalized),
    )
  }, [keyword, list])

  return (
    <div className={styles['contracts']}>
      <div className={styles['contracts__search']}>
        <Input
          value={keyword}
          onChange={(event) => setKeyword(event.target.value)}
          placeholder="Search indexed documents"
          suffix={<img src={IconSearch} alt="search" />}
        />
      </div>

      <div className={styles['contracts__title']}>
        {`Indexed documents (${filteredList.length})`}
      </div>

      <div className={styles['contracts__list']}>
        {filteredList.map((item) => (
          <ContractItem key={item.document_id} item={item} />
        ))}
      </div>
    </div>
  )
}