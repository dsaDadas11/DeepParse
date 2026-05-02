import IconCompleted from '@/assets/chat/completed.svg'
import IconFile from '@/assets/chat/file.svg'
import IconSelectFile from '@/assets/chat/select-file.svg'
import { RightOutlined } from '@ant-design/icons'
import { Button, Checkbox } from 'antd'
import { useState } from 'react'
import styles from './select-file.module.scss'

export default function ChooseFile(props: {
  list?: API.Document[]
  onSubmit?: (list: API.Document[]) => void
}) {
  const { list, onSubmit } = props

  const [selectedFileList, setSelectedFileList] = useState<string[]>([])

  return (
    <div className={styles['select-file']}>
      <div className={styles['select-file__header']}>
        <img className={styles['icon']} src={IconSelectFile} />
        Indexed documents: {list?.length ?? 0}
      </div>

      <Checkbox.Group
        className={styles['select-file__content']}
        value={selectedFileList}
        onChange={(value) => setSelectedFileList(value as string[])}
      >
        {list?.map((file) => (
          <Checkbox key={file.document_id} value={file.document_id}>
            <img className={styles['icon']} src={IconFile} />
            <div className={styles['name']} title={file.document_name}>
              {file.document_name}
            </div>
          </Checkbox>
        ))}
      </Checkbox.Group>

      <div className={styles['select-file__footer']}>
        <div>
          Selected <b>{selectedFileList.length}</b> documents
        </div>

        <Button
          type="primary"
          size="small"
          disabled={selectedFileList.length === 0}
          onClick={() => {
            onSubmit?.(
              list?.filter((file) =>
                selectedFileList.includes(file.document_id),
              ) ?? [],
            )
          }}
        >
          Apply
        </Button>
      </div>
    </div>
  )
}

function SelectFileSearching() {
  return (
    <div className={styles['select-file-searching']}>
      <div className={styles['icon']}></div>
      <div className={styles['title']}>Searching indexed evidence...</div>
    </div>
  )
}
ChooseFile.Searching = SelectFileSearching

function SelectFileComplete(props: {
  contractsLength: number
  citationsLength: number
  onClick?: () => void
}) {
  const { contractsLength, citationsLength, onClick } = props

  return (
    <div className={styles['select-file-complete']} onClick={onClick}>
      <img className={styles['icon']} src={IconCompleted} />
      <div className={styles['title']}>Evidence ready</div>
      <div className={styles['desc']}>
        {citationsLength ?? 0} citations across {contractsLength ?? 0} documents
      </div>
      <RightOutlined className={styles['arrow']} />
    </div>
  )
}

ChooseFile.Complete = SelectFileComplete
