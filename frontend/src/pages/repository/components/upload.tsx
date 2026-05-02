import * as api from '@/api'
import IconUpload from '@/assets/repository/upload.svg'
import { Upload, UploadFile, UploadProps } from 'antd'
import { forwardRef, useImperativeHandle, useState } from 'react'
import styles from './upload.module.scss'

export type RepositoryUploadRef = {
  submit: () => Promise<void>
}

async function waitForUploadTask(taskId: string) {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const { data } = await api.repository.getUploadTask({ task_id: taskId })
    const task = data.task
    if (task.status === 'success') {
      return task
    }
    if (task.status === 'failed') {
      throw new Error(task.error || task.message || 'Indexing failed.')
    }
    await new Promise((resolve) => setTimeout(resolve, 1500))
  }
  throw new Error('Indexing is still running. Please refresh the repository list later.')
}

export default forwardRef<RepositoryUploadRef, UploadProps>(
  function RepositoryUpload(props: UploadProps, ref) {
    const maxUploadSizeBytes = 5 * 1024 * 1024
    const { ...otherProps } = props
    const [fileList, setFileList] = useState<UploadFile[]>([])

    useImperativeHandle(ref, () => {
      return {
        submit: async () => {
          let hasError = false
          const errors: Error[] = []

          for (const file of fileList) {
            if (file.status === 'done') continue

            setFileList((prev) =>
              prev.map((item) => {
                if (item.uid === file.uid) {
                  return {
                    ...item,
                    status: 'uploading',
                  }
                }
                return item
              }),
            )

            try {
              if ((file.size ?? 0) > maxUploadSizeBytes) {
                throw new Error('File size cannot exceed 5 MB.')
              }

              const { data } = await api.repository.upload({
                files: file.originFileObj as File,
              })
              const taskId = data.tasks?.[0]?.task_id
              if (taskId) {
                await waitForUploadTask(taskId)
              }

              setFileList((prev) =>
                prev.map((item) => {
                  if (item.uid === file.uid) {
                    return {
                      ...item,
                      status: 'done',
                      url: '#',
                    }
                  }
                  return item
                }),
              )
            } catch (error) {
              hasError = true
              const nextError =
                error instanceof Error ? error : new Error('Unknown upload error')
              errors.push(nextError)
              setFileList((prev) =>
                prev.map((item) => {
                  if (item.uid === file.uid) {
                    return {
                      ...item,
                      status: 'error',
                      response: nextError.message,
                    }
                  }
                  return item
                }),
              )
            }
          }

          if (hasError) {
            window.$app.message.error(errors?.[0]?.message)
            throw new Error(errors?.[0]?.message)
          }

          window.$app.message.success('Upload completed.')
        },
      }
    })

    return (
      <div className={styles['repository-upload']}>
        <Upload.Dragger
          {...otherProps}
          showUploadList={false}
          maxCount={10}
          fileList={fileList}
          onChange={(info) => setFileList(info.fileList)}
        >
          <img src={IconUpload} alt="" />
          <p
            className="ant-upload-text"
            style={{
              color: '#666',
            }}
          >
            Drag files here or <span style={{ color: '#409EFF' }}>click to upload</span>
          </p>
        </Upload.Dragger>

        <p className={styles['repository-upload__desc']}>
          Supports single or batch upload. The size of each file must stay under 5 MB.
        </p>

        <Upload
          fileList={fileList}
          onChange={(info) => setFileList(info.fileList)}
        />
      </div>
    )
  },
)
