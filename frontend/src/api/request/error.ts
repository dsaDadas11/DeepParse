import { AxiosResponse } from 'axios'

export class ResponseError extends Error {
  response: AxiosResponse<unknown> | undefined

  constructor(message: string, response?: AxiosResponse<unknown>) {
    super(message)
    this.response = response
  }
}
