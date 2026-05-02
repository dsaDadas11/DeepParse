import IconSendThunder from '@/assets/component/send-thunder.svg'
import { Button, Input, Space } from 'antd'
import classNames from 'classnames'
import { PropsWithChildren, useState } from 'react'
import './index.scss'

export default function ComSender(
  props: PropsWithChildren<{
    className?: string
    loading?: boolean
    onSend?: (value: string) => void | Promise<void>
    onContract?: () => void
    sessionId?: string
  }>,
) {
  const { className, onSend, loading, ...rest } = props
  const [value, setValue] = useState('')

  async function send() {
    if (loading) return
    if (!value.trim()) return
    await onSend?.(value)
    setValue('')
  }

  return (
    <div className={classNames('com-sender', className)} {...rest}>
      <Input.TextArea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Ask a legal question based on your uploaded materials"
        autoSize={{ minRows: 2 }}
        autoFocus
      />

      <div className="com-sender__actions">
        <Space className="com-sender__actions-left" size={12} />

        <Space className="com-sender__actions-right" size={12}>
          <Button
            className="com-sender__action--send"
            variant="solid"
            color="primary"
            shape="round"
            onClick={send}
            loading={loading}
          >
            Send
            <img src={IconSendThunder} />
          </Button>
        </Space>
      </div>
    </div>
  )
}