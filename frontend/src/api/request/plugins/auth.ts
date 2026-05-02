import { router } from '@/router/routes'
import { ResponseError } from '../error'
import { IRequestPlugin } from './plugin'
import { MESSAGE_KEY } from './service'

export const authPlugin: IRequestPlugin = {
  install(instance) {
    instance.interceptors.response.use(
      (response) => response,
      (error) => {
        const response = error.response
        if (!response) return Promise.reject(error)

        if (response.status === 461) {
          const message = response?.data?.[MESSAGE_KEY] || 'Please upload documents first.'
          router.navigate('/repository')
          return Promise.reject(new ResponseError(message, response))
        }

        return Promise.reject(error)
      },
    )
  },
}