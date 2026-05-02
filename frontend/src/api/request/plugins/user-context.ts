import { getViewerId } from '@/utils/viewer-id'
import { IRequestPlugin } from './plugin'

export const userContextPlugin: IRequestPlugin = {
  postinstall(instance) {
    instance.interceptors.request.use((config) => {
      const viewerId = getViewerId()
      const headers = config.headers

      if (headers && typeof headers.set === 'function') {
        headers.set('X-User-Id', viewerId)
      } else {
        config.headers = {
          ...headers,
          'X-User-Id': viewerId,
        }
      }

      return config
    })
  },
}
