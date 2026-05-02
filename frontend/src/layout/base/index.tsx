import { DatabaseOutlined, EditOutlined } from '@ant-design/icons'
import { deviceState } from '@/store/device'
import { useNavigate } from 'react-router-dom'
import { useSnapshot } from 'valtio'
import { Background } from './background'
import './index.scss'
import { Nav } from './nav'

const TITLE = import.meta.env.VITE_TITLE

export function BaseLayout({ children }: { children?: React.ReactNode }) {
  const navigate = useNavigate()
  const device = useSnapshot(deviceState)

  return (
    <div className="base-layout">
      <div className="base-layout__sidebar">
        <div className="base-layout__logo">
          <span className="title">{TITLE}</span>
        </div>

        <div className="base-layout__sidebar-main scrollbar-style">
          <div className="base-layout__sidebar-main-content">
            <div
              className="base-layout__nav-header"
              onClick={() => (device.chatting ? null : navigate('/'))}
            >
              <span className="base-layout__nav-header-icon">
                <EditOutlined />
              </span>
              <span className="base-layout__nav-header-title">New Legal Chat</span>
            </div>

            <Nav />

            <div
              className="base-layout__nav-header"
              onClick={() => (device.chatting ? null : navigate('/repository'))}
            >
              <span className="base-layout__nav-header-icon">
                <DatabaseOutlined />
              </span>
              <span className="base-layout__nav-header-title">Legal Knowledge Base</span>
            </div>
          </div>
        </div>
      </div>

      <div className="base-layout__content">{children}</div>

      <Background />
    </div>
  )
}
