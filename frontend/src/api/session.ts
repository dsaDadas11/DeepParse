import { AxiosRequestConfig } from 'axios'
import { request } from './request'

export function list(
  params?: Record<string, unknown>,
  options?: AxiosRequestConfig,
) {
  return request.get<{
    sessions: API.Session[]
  }>(`/get_sessions`, {
    ...options,
    params,
  })
}

export function detail(
  params: {
    session_id: string
  },
  options?: AxiosRequestConfig,
) {
  return request.get<
    {
      created_at: string
      message_id: string
      session_id: string
      user_question: string
      model_answer: string
      think?: string
      documents?: string
      recommended_questions?: string
    }[]
  >(`/get_messages`, {
    ...options,
    params,
  })
}

export function create(
  params?: Record<string, never>,
  options?: AxiosRequestConfig,
) {
  return request.post<
    API.Result<{
      session_id: string
    }>
  >(`/create_session`, params, options)
}

export function chat(
  params: {
    id: string
    message: string
  },
  options?: AxiosRequestConfig,
) {
  const { id, ..._params } = params
  return request.post<ReadableStream>(
    '/chat_on_docs',
    {
      ..._params,
    },
    {
      headers: {
        Accept: 'text/event-stream',
      },
      responseType: 'stream',
      adapter: 'fetch',
      loading: false,
      params: {
        session_id: id,
      },
      ...options,
    },
  )
}
