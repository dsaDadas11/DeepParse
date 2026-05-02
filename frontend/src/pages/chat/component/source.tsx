import IconFile from '@/assets/chat/file.svg'
import styles from './source.module.scss'

export default function Source(props: { list: API.Document[] }) {
  const { list } = props

  return (
    <div className={styles['source']}>
      <div className={styles['source__title']}>Active sources</div>

      <div className={styles['source__list']}>
        {list.map((source) => (
          <div className={styles['source__item']} key={source.document_id}>
            <img className={styles['icon']} src={IconFile} />
            <span className={styles['name']} title={source.document_name}>
              {source.document_name}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
