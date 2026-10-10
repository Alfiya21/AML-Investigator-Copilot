# AML Investigator Copilot

An assistant that helps an AML analyst work through a flagged account. You pick an account and ask something like "why was this flagged, and what do the rules say?". A LangGraph agent then:

1. calls tools over MCP to get the account's risk score, the SHAP reasons behind it, its profile and recent transactions
2. searches public AML guidance (text, tables and figure captions) with hybrid retrieval and a reranker
3. drafts a short case summary where every fact carries a source tag
4. checks each claim against its sources and re-drafts once if something isn't supported

It is decision support. The draft says "indicators consistent with", never "guilty", and the unsupported-claim count is shown to the analyst instead of hidden.

All transaction data is synthetic. Nothing here uses real customer data.

## How it fits together

```
question + account
      |
  investigate  --(MCP, stdio)-->  score_account, explain_flag,
      |                           get_account_profile, get_recent_transactions
      v
   retrieve     query rewrite -> FAISS + BM25 -> RRF -> cross-encoder rerank
      |
      v
    draft  <-----------+
      |                | unsupported claims (max 2 drafts)
      v                |
    verify ------------+
```

- `src/aml_copilot/transactions` - synthetic data, SQLite, features, XGBoost, SHAP
- `src/aml_copilot/mcp_server` - the MCP tool server
- `src/aml_copilot/ingestion` and `retrieval` - PDF parsing, chunking, captions, hybrid search
- `src/aml_copilot/agent` - the LangGraph graph and its nodes
- `src/aml_copilot/api` - FastAPI app and a plain web page
- `prompts/` - versioned prompts (`v1_*.txt`, switch with `PROMPT_VERSION`)

## Setup

```
python -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env        # add your OpenAI key
```

Put a few public AML PDFs (FATF or FinCEN guidance, typology reports) in `data/raw_pdfs/`.

## Run

```
make data      # synthetic accounts + transactions -> data/transactions.db
make train     # XGBoost model -> models/
make index     # parse PDFs, caption figures, build the search index
make api       # http://127.0.0.1:8000
```

`python scripts/build_index.py --no-vision` skips figure captioning if you want to test without spending on the vision model.

Evaluation and checks:

```
make test      # unit and integration tests (no API key needed)
make eval      # retrieval hit rate / MRR, faithfulness, hallucination rate
make bias      # flag rate, recall and false positive rate by segment and region
```

For `make eval`, copy `eval/questions.sample.json` to `eval/questions.json` and write about 50 questions of your own, each with `expected_phrases` that appear in the passage that answers it.

## Results

## Results

The following results are from the project's previous local evaluation run. Model metrics were measured on a **synthetic test dataset** and should not be interpreted as evidence of real-world AML detection performance.

| **What** | **Value** |
|---|---|
| Model ROC-AUC (synthetic test set) | 0.9924 |
| Model PR-AUC (synthetic test set) | 0.9659 |
| Model recall at chosen threshold | 0.7812 (78.12%) |
| Model precision at chosen threshold | 1.0000 (100%) |
| Test set size | 800 accounts |
| Positive cases in test set | 32 |
| False positives / false negatives | 0 / 7 |
| Retrieval hit rate@5 / MRR | Not measured yet |
| Faithfulness / hallucination rate | Not measured yet |
| p50 / p95 latency per investigation | Not measured yet |
| Average cost per investigation | Not measured yet; local LLM setup and end-to-end testing pending |

### Evaluation notes

- The classification metrics above come from the previous synthetic-data test run.
- The selected classification threshold should be documented alongside the model evaluation configuration.
- Retrieval quality, evidence faithfulness, hallucination rate, and investigation latency require dedicated evaluation runs.
- Cost per investigation has not been measured. The planned Ollama configuration is intended to run inference locally without paid OpenAI API credits, but local compute still has hardware and electricity costs.
- These results are preliminary and do not establish production readiness or regulatory suitability.
## Known limits

- Page images are captioned by a vision LLM and indexed as text. It is not image-embedding retrieval.
- The verify step uses an LLM, so it can miss problems too. Treat it as a filter, not a guarantee.
- The agent can only call the four tools in `mcp_server/server.py`, and the account id is forced to the one in the request.
- With `mcp` 2.x the server code needs changes, so `requirements.txt` pins `mcp<2`.
