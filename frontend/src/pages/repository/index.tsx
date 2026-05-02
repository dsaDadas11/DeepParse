import * as api from '@/api'
import IconDelete from '@/assets/repository/action/delete.svg'
import IconSearch from '@/assets/repository/search.svg'
import { PlusOutlined } from '@ant-design/icons'
import { useRequest } from 'ahooks'
import { Button, Input, Modal, Popconfirm, Space, Table } from 'antd'
import { ColumnsType } from 'antd/es/table'
import dayjs from 'dayjs'
import { useMemo, useRef, useState } from 'react'
import { FileIcon } from './components/file-icon'
import RepositoryUpload, { RepositoryUploadRef } from './components/upload'
import styles from './index.module.scss'

type RepositoryRow = API.Repository & {
  id: number
  suffix: FileIcon
}

export default function RepositoryPage() {
  const [keyword, setKeyword] = useState('')

  const { data = [], refresh } = useRequest(async () => {
    const { data } = await api.repository.list()
    return (data || []).map(
      (item, index) =>
        ({
          ...item,
          id: index + 1,
          suffix: item.file_name.split('.').pop() as FileIcon,
        }) satisfies RepositoryRow,
    )
  })

  const filteredData = useMemo(() => {
    const normalized = keyword.trim().toLowerCase()
    if (!normalized) return data
    return data.filter((item) => item.file_name.toLowerCase().includes(normalized))
  }, [data, keyword])

  const columns = useMemo<ColumnsType<RepositoryRow>>(
    () => [
      {
        title: 'Material',
        dataIndex: 'file_name',
        width: 260,
        render(value, row) {
          return (
            <div className={styles['repository-page__file-name']} title={value}>
              <FileIcon className={styles['icon']} suffix={row.suffix} />
              {value}
            </div>
          )
        },
      },
      {
        title: 'Updated At',
        dataIndex: 'updated_at',
        width: 220,
        render(value) {
          return dayjs(value).format('MM/DD/YYYY HH:mm:ss')
        },
      },
      {
        title: 'Operation',
        dataIndex: 'action',
        width: 120,
        render(_, row) {
          return (
            <Space>
              <Popconfirm
                title="Remove this legal material from the legal knowledge base?"
                onConfirm={async () => {
                  await api.repository.remove({ file_name: row.file_name })
                  refresh()
                }}
              >
                <Button
                  color="default"
                  variant="text"
                  shape="circle"
                  size="small"
                >
                  <img src={IconDelete} />
                </Button>
              </Popconfirm>
            </Space>
          )
        },
      },
    ],
    [refresh],
  )

  const [openUpload, setOpenUpload] = useState(false)
  const uploadRef = useRef<RepositoryUploadRef>(null)
  const [uploading, setUploading] = useState(false)

  return (
    <div className={styles['repository-page']}>
      <div className={styles['repository-page__header']}>
        <div className={styles['title']}>Legal Knowledge Base</div>
        <div className={styles['desc']}>
          Upload legal materials, build the index, and cite them in legal chat.
        </div>
      </div>

      <div className={styles['repository-page__body']}>
        <div className={styles['header']}>
          <Input
            value={keyword}
            onChange={(event) => setKeyword(event.target.value)}
            placeholder="Search legal materials"
            prefix={<img src={IconSearch} />}
            style={{ width: 240 }}
          />

          <Button type="primary" onClick={() => setOpenUpload(true)}>
            <PlusOutlined />
            Upload materials
          </Button>
        </div>

        <Table<RepositoryRow>
          rowKey="id"
          columns={columns}
          dataSource={filteredData}
          pagination={false}
        />
      </div>

      <Modal
        title="Upload legal materials"
        open={openUpload}
        width={400}
        destroyOnClose
        onCancel={() => {
          if (uploading) return
          setOpenUpload(false)
        }}
        onOk={async () => {
          setUploading(true)
          try {
            await uploadRef.current?.submit()
            setOpenUpload(false)
            refresh()
          } finally {
            setUploading(false)
          }
        }}
      >
        <RepositoryUpload beforeUpload={() => false} ref={uploadRef} />
      </Modal>
    </div>
  )
}