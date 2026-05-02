declare namespace API {
  interface Session {
    created_at: string
    session_id: string
    session_name: string
    updated_at: string
    // user_id: string
  }

  interface ChatItem {
    id: number
    role: import('@/configs').ChatRole
    type: import('@/configs').ChatType
    created_at?: string
    loading?: boolean
    error?: string
    content?: string
    think?: string

    documents?: Document[]
    reference?: Reference[]
    recommended_questions?: string[]
    answer_audit?: AnswerAudit
    retrieval_trace?: RetrievalTrace
  }

  interface Document {
    document_id: string
    document_name: string
    content_with_weight: string
  }

  interface Reference {
    id: string
    rank?: number
    display_rank?: number
    source_rank?: number
    best_rank?: number
    merge_hit_count?: number
    citation_index?: number
    citation_marker?: string
    rule_citation_key?: string
    document_id: string
    chunk_id?: string
    document_name: string
    content_with_weight: string
    positions: number[][]
    page_num?: number | null
    page_num_int?: number[]
    company?: string
    report_period?: string
    report_type?: string
    source?: string
    similarity?: number
    term_similarity?: number
    vector_similarity?: number
    best_similarity?: number
    fusion_score?: number
    location_label?: string
  }

  interface RetrievalTrace {
    question?: string
    standalone_query?: string
    planned_queries?: string[]
  }

  interface AnswerAudit {
    mode?: string
    rule_reason?: string | null
    citation_indices?: number[]
    supporting_chunks?: Reference[]
    supporting_documents?: string[]
    retrieval_trace?: RetrievalTrace
  }
}
