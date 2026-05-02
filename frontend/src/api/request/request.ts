import axios, { AxiosRequestConfig } from 'axios'
import { authPlugin } from './plugins/auth'
import { errorToastPlugin } from './plugins/error-toast'
import { loadingPlugin } from './plugins/loading'
import { installPlugins } from './plugins/plugin'
import { repeatPlugin } from './plugins/repeat'
import { servicePlugin } from './plugins/service'
import { userContextPlugin } from './plugins/user-context'

export function createRequest(configs: AxiosRequestConfig = {}) {
  const instance = axios.create(configs)

  installPlugins(instance, [
    userContextPlugin,
    authPlugin,
    servicePlugin,
    loadingPlugin,
    repeatPlugin,
    errorToastPlugin,
  ])

  return instance
}
