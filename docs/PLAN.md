# Implementation plan: AML Investigator Copilot

About 8 weeks at 10-12 hours a week. Each week ends with something that runs and a number you can write down.

## Read this first

The code in this repo is a working starting point. What makes the project yours (and defensible in an interview) is that you understand every file. So:

- Don't paste everything in one sitting. Type or at least read each file in the order below, run its check, and commit. The commit history should look like a project that grew over weeks, because it did.
- Change things as you go: rename features, adjust the prompts, try a different chunk size. Every change you can explain is worth more than a clean copy.
- After each week, close the files and explain what you built out loud, as if to an interviewer.

What has and hasn't been tested so far:

| Tested offline (21 tests pass) | Not run yet, you need to do this |
|---|---|
| Synthetic data, features, XGBoost, SHAP | Anything that calls OpenAI (captioning, drafting, verify, eval judge) |
| MCP server over stdio, tool schemas | Downloading the embedding and reranker models |
| LangGraph flow incl. the re-draft loop (LLM faked) | Indexing real PDFs |
| FastAPI endpoints, metrics store | Docker build, AWS, Databricks |
| PDF parsing, chunking, hybrid index (embedder faked) | Prompt quality on real documents |

Expect to fix small things on the first live run. That is normal and useful.

---

## Week 0: setup (1 evening)

1. Create the repo, copy this folder in, `make install`, `cp .env.example .env`, add your key.
2. Set a spending limit on your OpenAI account before anything else.
3. Choose 3 to 5 public PDFs and put them in `data/raw_pdfs/`. Pick ones that contain at least one table and one chart. FATF typology reports and FinCEN advisories are good. Check each document's terms before you redistribute it; you only need them locally, so don't commit them (they are git-ignored).
4. `make test` should pass on your machine.

## Week 1: transactions, model, explanations

Files: `transactions/synth.py`, `db.py`, `features.py`, `train.py`, `scorer.py`, `explain.py`.

```
make data && make train
```

Do:
- Read `synth.py` and understand the three patterns and why the look-alike accounts exist. Try setting `lookalike_rate=0` and see what happens to the AUC. Write down why a perfect score is a warning sign, not a win.
- Add two features of your own (for example the ratio of night-time transactions, or the number of distinct days with activity). Retrain and compare PR-AUC.
- Look at `explain.top_reasons` for a few flagged accounts and check the reasons make sense against the pattern that was injected.

Done when: you have a PR-AUC, recall and precision on the held-out set, and you can explain why the threshold is picked on a separate validation split.

Be ready to explain: class imbalance (`scale_pos_weight`), why accuracy is useless here, what SHAP values mean for one prediction.

## Week 2: documents, hybrid retrieval, evaluation harness first

Files: `ingestion/pdf_parser.py`, `chunking.py`, `retrieval/*`, `evaluation.py`, `scripts/build_index.py`, `scripts/run_eval.py`.

```
python scripts/build_index.py --no-vision
```

Do:
- Write 50 questions in `eval/questions.json` before tuning anything. Each needs `expected_phrases`: words that appear in the passage that answers it. Mix easy, wording-mismatch ("moving money fast" vs "rapid movement of funds") and table questions.
- `python scripts/run_eval.py --retrieval-only` to get hit rate@5 and MRR.
- Run experiments, one change at a time, and keep a small results table in the README:
  - dense only vs BM25 only vs hybrid (temporarily comment out one side in `HybridIndex.search`)
  - with and without the reranker
  - chunk size 500 / 900 / 1500

Done when: you have a table showing what hybrid and the reranker each add (or don't). A result of "the reranker added nothing on my documents" is a fine finding if you measured it.

Be ready to explain: what reciprocal rank fusion does, why BM25 still matters, why the reranker runs on 20 candidates and not the whole corpus.

## Week 3: multimodal ingestion

Files: `ingestion/captioning.py`, plus `looks_visual` in `pdf_parser.py`.

```
python scripts/build_index.py
```

Do:
- Open `data/captions.json` and read the captions. Are they accurate? Fix the caption prompt in `prompts/v1_caption.txt` if they aren't (save the new one as `v2_caption.txt` and set `PROMPT_VERSION=v2`, so you keep the old one).
- Check which pages were marked visual. Tune the thresholds in `looks_visual` for your PDFs.
- Add 10 to 15 questions that can only be answered from a table or a chart, then compare hit rate with and without figure and table chunks.

Done when: you can show the questions that text-only retrieval misses and the multimodal index gets. That comparison is the core claim of the "multimodal" part, so measure it.

Honest framing for the resume: "vision-LLM captioning of page images, indexed alongside text and tables". Don't call it image-embedding search unless you add CLIP.

## Week 4: MCP tools and the agent

Files: `mcp_server/server.py`, `agent/mcp_client.py`, `state.py`, `nodes.py`, `graph.py`.

Do:
- Run the server on its own and inspect it with the MCP inspector (`npx @modelcontextprotocol/inspector python -m aml_copilot.mcp_server.server`). Call each tool by hand.
- Read `nodes.investigate`. Notice that the account id is overwritten, and think about why (prompt injection or a confused model asking for another customer's data).
- Add one more tool of your own, for example `get_counterparty_summary`, and see the agent use it.
- Run `make api` and investigate five accounts. Read every answer critically.

Done when: five investigations run end to end and you can point to one where the agent chose a tool you didn't expect.

Be ready to explain: what MCP gives you over calling Python functions directly (a standard, separately runnable tool boundary), what stdio transport means, and what could go wrong if tools could run arbitrary SQL.

## Week 5: verification, evaluation of answers, observability

Files: `nodes.verify`, `observability.py`, the `/metrics` endpoint, the page in `api/static`.

Do:
- `python scripts/run_eval.py` for faithfulness and hallucination rate on document questions.
- Hand-check 20 verify decisions. Where does the LLM judge disagree with you? Write down the false-pass and false-fail rate. This is your honest "how reliable is the judge" number.
- Read `/metrics` after 20+ runs. Which step costs the most? Which is slowest? Try a cheaper model for `queries` and see whether quality drops.

Done when: README table has faithfulness, hallucination rate, p50/p95 latency and cost per investigation, all measured.

## Week 6: bias check, CI, Docker, versioning

Do:
- `make bias`. Look at recall and false positive rate per segment and region. Small groups are noisy, so say so. If one group is flagged much more often, check whether that's the data (small businesses really do use more cash in this generator) or the model.
- Push to GitHub and confirm the Actions run goes green.
- `docker build -t aml-copilot .` then run it with `-v $(pwd)/data:/app/data -v $(pwd)/models:/app/models --env-file .env -p 8000:8000`. (The Dockerfile has not been built yet, so expect to fix something.)
- Versioning: the index folder name comes from a hash of the content, prompts carry `v1_`, `v2_`, and models carry `MODEL_VERSION`. Write a short section in the README on how you'd roll back.

Done when: CI is green, the container serves the UI, and the bias table is in the README with caveats.

## Week 7: cloud and data platform (only claim what you actually do)

AWS, simplest path:
1. Push the image to ECR.
2. Run it on App Runner or a small EC2 instance. Put `OPENAI_API_KEY` in Secrets Manager or SSM, not in the image.
3. Keep PDFs and the built index in S3 and sync them at start-up, or bake the index into a volume.
4. Set a billing alarm first. Stop or delete everything when you finish the demo.

Databricks (Free Edition):
1. Upload the `transactions` table as a Delta table.
2. Re-implement `build_features` in PySpark (groupBy plus window functions for the pass-through feature).
3. Check that the PySpark features match the pandas ones on the same accounts. Write down any differences.
4. Only after that is it honest to write "reimplemented the feature pipeline in PySpark on Databricks".

Azure: skip it unless you really deploy something there. One cloud done properly beats two done halfway.

## Week 8: optional LoRA, polish, resume

LoRA (optional, needs a GPU such as free Colab):
- Collect 200 to 500 drafts that passed verification with zero unsupported claims, plus their contexts, as training examples.
- Fine-tune a small open instruct model with `peft` and `trl`.
- Run the same faithfulness eval on the base model, the tuned model, and the API model. Report all three.
- A small tuned model will often lose to the API model. That is a legitimate result. The claim you can make is "fine-tuned and evaluated", not "improved", unless the numbers say so.

Polish:
- Record a 2 minute screen capture of one investigation.
- Make the README results table complete. Remove anything you didn't do.
- Pin the repo on GitHub and add the live demo link.

---

## When you're done, send me

1. The GitHub link.
2. The filled-in results table.
3. A list of what you built and what you skipped.

I'll rewrite the resume around it and re-score it against the Mastercard JD.

## Resume bullets (fill in after you have the numbers)

Use these shapes, and delete any line you can't back up:

- Built an agentic RAG assistant for AML alert investigation (LangGraph, MCP, FastAPI) that retrieves from [N] regulatory PDFs and calls model, SQL and SHAP tools, drafting cited case summaries.
- Implemented hybrid retrieval (FAISS + BM25 with reciprocal rank fusion and a cross-encoder reranker); measured hit rate@5 of [X] and MRR of [Y] on a [N]-question test set.
- Added vision-LLM captioning and table extraction so figure and table questions are answerable; improved hit rate on those questions from [A] to [B].
- Built a claim-level verification step and evaluation harness: faithfulness [X], hallucination rate [Y]; logged latency, tokens and cost per run (p95 [Z] s, $[C] per investigation).
- Trained an XGBoost account-risk model on [N] synthetic accounts (PR-AUC [X]) and checked flag rates and recall across customer segments.
- Containerised the service with Docker and CI on GitHub Actions[, deployed on AWS].

## Questions you should be able to answer

- Why hybrid retrieval instead of vector search alone?
- How do you know the retrieval is good? What's your test set and what are its weaknesses?
- What does the verify step catch, and what does it miss?
- Why an agent with tools instead of one big prompt?
- What stops the agent from reading another customer's data?
- How would this change with real transaction data (privacy, label quality, drift)?
- Your model scores 0.99 AUC. Why shouldn't anyone be impressed by that?
- What would you build next?

## Common problems

| Problem | Likely cause and fix |
|---|---|
| `No module named mcp.server.fastmcp` | `mcp` 2.x installed. Run `pip install "mcp<2"`. |
| First `make index` is slow | The embedding and reranker models are downloading. It only happens once. |
| No tables found in a PDF | `pdfplumber` needs ruled tables. For borderless tables, try its `text` strategy or a layout model. |
| Every page captioned | Raise the image-size or drawing-count thresholds in `looks_visual`. |
| `OPENAI_API_KEY` errors in Docker | Pass `--env-file .env`. |
| Verify marks almost everything unsupported | Check that the draft's tags (T1, D2) match the context numbering, and try a stronger model for verify only. |
| XGBoost import error in a slim image | Install `libgomp1` (already in the Dockerfile). |
