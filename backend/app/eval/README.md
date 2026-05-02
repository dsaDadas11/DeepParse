# Legal RAG Rebuild And Eval

## Full rebuild from scratch

```bash
docker exec deepparse_api python /app/rebuild_user_corpus.py \
  --user-id <USER_ID> \
  --source-dir /app/service/core/storage/file/<USER_ID> \
  --reset-first \
  --continue-on-error
```

## Resume a stopped rebuild

```bash
docker exec deepparse_api python /app/rebuild_user_corpus.py \
  --user-id <USER_ID> \
  --source-dir /app/service/core/storage/file/<USER_ID> \
  --resume \
  --continue-on-error
```

## Legal corpus default source

The legal retrieval runtime loads the default corpus list from:

`/app/sample_data/pdf_list.txt`

(Repository path: `backend/app/sample_data/pdf_list.txt`)

## Run retrieval evaluation

```bash
docker exec deepparse_api python /app/eval/run_retrieval_eval.py \
  --user-id <USER_ID> \
  --cases /app/eval/resume_retrieval_benchmark_manual_v2.json \
  --output /app/eval/latest_retrieval_eval_report.json
```

## Run generation evaluation

```bash
docker exec deepparse_api python /app/eval/run_generation_eval.py \
  --user-id <USER_ID> \
  --cases /app/eval/resume_generation_benchmark_manual_v2.json \
  --output /app/eval/latest_generation_eval_report.json
```

## Build benchmark manifest

```bash
docker exec deepparse_api python /app/eval/build_benchmark_manifest.py \
  --user-id <USER_ID> \
  --retrieval-cases /app/eval/resume_retrieval_benchmark_manual_v2.json \
  --generation-cases /app/eval/resume_generation_benchmark_manual_v2.json \
  --retrieval-report /app/eval/latest_retrieval_eval_report.json \
  --generation-report /app/eval/latest_generation_eval_report.json \
  --output /app/eval/benchmark_manifest.json
```

## Benchmark focus (legal domain)

- statute article locating (`law_article_*`)
- version/effective-date targeting (`law_version_*`, `version_*`)
- guiding-case retrieval (`case_id_*`)
- procedure and form retrieval (`procedure_*`)
- contract template version targeting (`contract_*`)
- abstain behavior for out-of-corpus legal requests

## Naming compatibility note

Some benchmark files keep historical `resume_*` prefixes for compatibility with existing scripts.
Treat them as legal benchmark datasets based on their actual content.
