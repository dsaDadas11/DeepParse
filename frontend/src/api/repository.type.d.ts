declare namespace API {
  type Repository = {
    created_at: string
    file_name: string
    updated_at: string
    user_id: string
  }

  type UploadTask = {
    task_id: string
    user_id: string
    file_name: string
    status: 'pending' | 'running' | 'success' | 'failed'
    message: string
    created_at: string
    started_at?: string | null
    finished_at?: string | null
    indexed_chunks?: number | null
    error?: string | null
    retry_count?: number | null
  }
}
