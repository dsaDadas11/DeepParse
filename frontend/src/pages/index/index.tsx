import * as api from '@/api'
import { useMount } from 'ahooks'
import { useNavigate } from 'react-router-dom'

export default function Index() {
  const navigate = useNavigate()

  useMount(() => {
    void (async () => {
      try {
        const { data } = await api.session.create()
        navigate(`/chat/${data.session_id}`)
      } catch (error) {
        const message =
          error instanceof Error ? error.message : 'Failed to create session.'
        window.$app.message.error(message)
      }
    })()
  })

  return null
}
