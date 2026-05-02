import { AxiosRequestConfig } from 'axios'
import { request } from './request'

export function list(
  params?: Record<string, unknown>,
  options?: AxiosRequestConfig,
) {
  return request.get<API.Repository[]>('/get_files', {
    ...options,
    params,
  })
}

export function upload(params: { files: File }, options?: AxiosRequestConfig) {
  const form = new FormData()
  form.append('files', params.files)
  return request.post<{
    queued_files?: string[]
    tasks?: API.UploadTask[]
    total_files: number
  }>(`/upload_files`, form, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    ...options,
  })
}

export function getUploadTask(
  params: { task_id: string },
  options?: AxiosRequestConfig,
) {
  return request.get<{ task: API.UploadTask }>(
    `/upload_tasks/${encodeURIComponent(params.task_id)}`,
    options,
  )
}

export function retryUploadTask(
  params: { task_id: string },
  options?: AxiosRequestConfig,
) {
  return request.post<{ task: API.UploadTask }>(
    `/upload_tasks/${encodeURIComponent(params.task_id)}/retry`,
    undefined,
    options,
  )
}

export function remove(
  params: { file_name: string },
  options?: AxiosRequestConfig,
) {
  const { file_name, ..._params } = params
  return request.delete(`/delete_file/${encodeURIComponent(file_name)}`, {
    ...options,
    params: _params,
  })
}
