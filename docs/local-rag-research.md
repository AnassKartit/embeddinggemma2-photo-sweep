# Local RAG and vector store research for "Ask your files"

Research date: 12 October 2026. Scope: open-source projects to learn from, the best embedded vector store for the Mac app
(Python) and the phone app (Flutter iOS/Android), and ideas worth copying.

How this was done: `gh search repos` / `gh api` (stars, last push, license, READMEs and source files read directly),
pub.dev API (package versions, publishers, 30-day downloads), and web search. Star counts and dates are from the GitHub
API on 2026-10-12. **One small benchmark was run on this Mac** (section 2.3); every other number is quoted from its
source and labelled *vendor* (published by the project itself) or *third party*.

## TL;DR

- **The closest prior art is [DigUp](https://github.com/ARahim3/DigUp)** (1,721 stars, MIT, macOS, released days after
  EmbeddingGemma 2). It does exactly "search your Mac by meaning" with EmbeddingGemma 2, and its whole index is **one SQLite
  file**: float16 vectors in a BLOB column scored by brute force in memory, plus an FTS5 keyword index. No vector database.
  Read its `Searcher.swift` before writing ours.
- **At our scale a vector index is not the bottleneck.** A big Downloads folder is thousands of files, so 5k to 50k vectors.
  Measured on this M4 Pro: numpy brute force over 50,000 x 256 vectors takes **1.4 ms** per query (median). What matters is
  metadata (kind, dates), keyword search, one portable file, and phone support.
- **Mac now: SQLite (Python stdlib) + FTS5 + numpy brute force**, vectors as BLOBs at 256 dims (keep 768 if cheap). Replaces
  the `.npz` + `.json` cache with one file that also answers "latest" and exact-word queries.
- **Phone: the same SQLite schema, with [sqlite-vec](https://github.com/asg017/sqlite-vec) (`vec0`) for KNN with metadata
  filters**, through `package:sqlite3` or [`flutter_edge_ai_sqlite`](https://pub.dev/packages/flutter_edge_ai_sqlite).
  Google's own EmbeddingGemma 2 sample ([MediaPipe `SemanticRetriever`](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/semantic_retriever))
  also uses a SQLite vector store on iOS and Android. Upgrade path if the corpus grows past ~50k vectors:
  [Qdrant Edge](https://qdrant.tech/edge/) (`qdrant_edge` on pub.dev, HNSW, payload filters, still beta).
- **Ideas to copy:** (1) DigUp's fusion: per-query z-scored cosine with a gate plus a keyword-coverage bonus (it beat
  reciprocal rank fusion on their evals); (2) parse "latest / newest / last year" into a date slot and sort the gated
  candidates by date instead of trusting the vector; (3) PDFs indexed per page and ranked by their best page, with citations
  and an opt-in small local LLM that answers only from retrieved evidence.

---

## 1. Open-source projects

### 1.1 The 15 most relevant

"Images" = does it embed images (not only OCR). Model = embedding model actually used.

| # | Project | Stars | Last push | License | Platform | Embedding model | Images | Store |
|---|---|---:|---|---|---|---|---|---|
| 1 | [ARahim3/DigUp](https://github.com/ARahim3/DigUp) | 1,721 | 2026-10-12 | MIT | macOS (Swift) | **EmbeddingGemma 2** (ggml 8-bit, llama.cpp Metal) | yes, + PDF, docs, audio, video, code | one SQLite file: f16 BLOB vectors, brute force in memory + FTS5 |
| 2 | [pdstn12/Recall](https://github.com/pdstn12/Recall) | 4 | 2026-10-12 | MIT | Linux (Rust, GTK 4) | **EmbeddingGemma 2** (llama.cpp CUDA/Vulkan/CPU) | yes | SQLite + FTS5 (DigUp's design rebuilt) |
| 3 | [ampersandru/VectorDash](https://github.com/ampersandru/VectorDash) | 15 | 2026-10-12 | MIT | Windows 11 (Python) | **EmbeddingGemma 2** GGUF + mmproj | yes, + audio, video | own compact encrypted vector cache, name index |
| 4 | [meangrinch/LocalLens](https://github.com/meangrinch/LocalLens) | 23 | 2026-10-12 | Apache-2.0 | Python web UI | **EmbeddingGemma 2** (70 to 1120 soft tokens), SigLIP 2 | yes, + video, duplicates | ChromaDB |
| 5 | [devexpert-io/local-photo-search](https://github.com/devexpert-io/local-photo-search) | 4 | 2026-10-08 | MIT | Python | **EmbeddingGemma 2** | yes (photos) | SQLite + sqlite-vec |
| 6 | [harpreetsahota204/cross_modal_similarity_search](https://github.com/harpreetsahota204/cross_modal_similarity_search) | 3 | 2026-10-09 | none | FiftyOne plugin | **EmbeddingGemma 2** | yes, + audio, video | LanceDB |
| 7 | [google-ai-edge/mediapipe-samples `semantic_retriever`](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/semantic_retriever) and [Gallery](https://github.com/google-ai-edge/gallery) | 2,859 / 24,935 | 2026-10-07 / 10-12 | Apache-2.0 | iOS, Android | **EmbeddingGemma 2** 440M `.litertlm` | yes | `SqliteVectorStore` (iOS + Android), `AppSearchVectorStore` (Android); Gallery adds metadata filters |
| 8 | [DenisovAV/flutter_edge_ai](https://github.com/DenisovAV/flutter_edge_ai) (ex `flutter_gemma`) | 638 | 2026-10-12 | MIT | Flutter, all 6 platforms | EmbeddingGemma 1, Gecko, SigLIP2 text, MiniLM; **EG2 "next week"** per maintainer ([issue #626](https://github.com/DenisovAV/flutter_edge_ai/issues/626), 2026-10-11) | not yet | sqlite-vec or Qdrant Edge, payload filters |
| 9 | [ente/ente](https://github.com/ente/ente) (Photos) | 29,365 | 2026-10-12 | AGPL-3.0 | **Flutter** + Rust | CLIP family, 512-d (MobileCLIP notebooks in `infra/ml/playground`) | yes, + faces, pets | USearch via flutter_rust_bridge, moving to their own Rust `ente_vecdb` |
| 10 | [mazzzystar/Queryable](https://github.com/mazzzystar/Queryable) | 2,989 | 2026-03-29 | MIT | iOS (Swift, Core ML) | Apple MobileCLIP S2 | yes (Photos library) | vectors saved on device, brute force |
| 11 | [allenv0/SCM](https://github.com/allenv0/SCM) | 475 | 2026-10-09 | MIT | macOS (Electron) | CLIP ViT-L/14@336 default, SigLIP-2 options | yes, + video frames, OCR, Whisper | JSON index + Float32 binary files; opt-in llama.cpp chat (Qwen3 1.7B) with citations |
| 12 | [yurijmikhalevich/rclip](https://github.com/yurijmikhalevich/rclip) | 1,016 | 2026-10-12 | MIT | CLI, all desktops | CLIP (ONNX) | yes | SQLite (`rclip/db.py`) |
| 13 | [shubham0204/OnDevice-RAG-Android](https://github.com/shubham0204/OnDevice-RAG-Android) | 201 | 2026-01-03 | Apache-2.0 | Android (Kotlin) | all-MiniLM-L6-v2 | no (PDF/DOCX text) | **ObjectBox** HNSW |
| 14 | [Mintplex-Labs/anything-llm](https://github.com/Mintplex-Labs/anything-llm) | 66,974 | 2026-10-12 | MIT | desktop + Docker | configurable (local default) | no (text docs) | **LanceDB** default ([docs](https://docs.anythingllm.com/setup/vector-database-configuration/local/lancedb)) |
| 15 | [immich-app/immich](https://github.com/immich-app/immich) | 116,107 | 2026-10-12 | AGPL-3.0 | self-hosted server + mobile client | CLIP / SigLIP2 models (configurable) | yes | Postgres + VectorChord ([docs](https://docs.immich.app/features/searching)) |

Also checked, less relevant:

| Project | Stars | Last push | License | Note |
|---|---:|---|---|---|
| [zylon-ai/private-gpt](https://github.com/zylon-ai/private-gpt) | 57,569 | 2026-10-09 | Apache-2.0 | text RAG API; `settings.yaml` default `vectorstore.database: qdrant` |
| [khoj-ai/khoj](https://github.com/khoj-ai/khoj) | 37,659 | 2026-08-02 | AGPL-3.0 | self-hosted "second brain"; Postgres + pgvector; server, not embedded |
| [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) | 77,365 | **2025-05-27** | MIT | LocalDocs: Nomic embeddings, SQLite + USearch (`gpt4all-chat/src/database.cpp`); repo looks dormant |
| [reorproject/reor](https://github.com/reorproject/reor) | 8,545 | **2025-05-13** | AGPL-3.0 | notes app, LanceDB + Transformers.js; dormant |
| [photoprism/photoprism](https://github.com/photoprism/photoprism) | 40,290 | 2026-10-12 | custom | embeddings in the repo are for faces (`internal/ai/vision/embeddings.go`); no CLIP-style text-to-image vector search found |
| [newdee/magpie](https://github.com/newdee/magpie) | 159 | 2026-10-08 | MIT | Spotlight-style launcher: SQLite FTS5 + multilingual-e5-small fused by RRF, ~1600-char overlapping chunks, doc ranked by best chunk, `dl` = newest downloads |
| [jiggersong/everythingsearch](https://github.com/jiggersong/everythingsearch) | 5 | 2026-08-17 | MIT | macOS: FTS5 + Chroma + RRF + file aggregation; an LLM turns the question into slots (`q`, `source`, `date_from`, `date_to`, `match_mode`) |
| [dev07060/mobile_rag_engine](https://github.com/dev07060/mobile_rag_engine) | 43 | 2026-10-08 | MIT | Flutter + Rust core: SQLite persistence, HNSW + BM25 with RRF, PDF text layer; ONNX text models only |
| [StarTrail-org/LEANN](https://github.com/StarTrail-org/LEANN) | 13,018 | 2026-10-09 | MIT | "97% less storage" (vendor) by recomputing embeddings at query time; wrong trade for us (an image embed is 85 ms) |
| [Obscyra-app/EmbeddingGemma2Swift](https://github.com/Obscyra-app/EmbeddingGemma2Swift) | 3 | 2026-10-06 | MIT | MLX Swift EG2 text + image + audio, validated on iPhone 16 Pro Max (author-reported); an iOS option besides LiteRT |
| [lattifai/awesome-EmbeddingGemma](https://github.com/lattifai/awesome-EmbeddingGemma) | 13 | 2026-10-12 | CC0 | list of EG2 apps and runtimes; lists DigUp and Recall as the apps |
| Msty, Rewind-style apps | n/a | n/a | closed | not open source, skipped |

### 1.2 What the group tells us

- **Everything built on EmbeddingGemma 2 in the last week is a "search your files" app**, and the most mature one (DigUp)
  chose plain SQLite, not a vector database. Recall copied that design for Linux.
- **The big "chat with your docs" apps** (AnythingLLM, PrivateGPT, Khoj, GPT4All) are text-only RAG with a server-style or
  general-purpose store (LanceDB, Qdrant, pgvector, USearch). None of them embeds images in the same space as text, which is
  the thing EmbeddingGemma 2 gives us for free.
- **Photo search** (Immich, Queryable, ente, rclip, SCM) is CLIP/SigLIP-based. Immich is server-side. **ente is the best
  reference for the Flutter app**: it runs CLIP on the phone inside a Flutter app and keeps vectors in a native (Rust)
  index next to its SQLite data, with exact and approximate modes (`bulkSearchNearestForFiles(..., exact:)`).
- **Mobile RAG on Flutter is converging on SQLite + sqlite-vec** (flutter_edge_ai_sqlite, Google's `SqliteVectorStore`),
  with Qdrant Edge as the fast native option and ObjectBox as the older mature option.

### 1.3 DigUp in detail (the one to study)

From `Sources/DigUpKit/Searcher.swift` (read 2026-10-12):

- Index = one SQLite file. Table `segments` holds one vector per picture, PDF page, document chunk, video frame and 30 s of
  sound (`vec` BLOB, 768 x float16). At search time all vectors are loaded into one matrix and every one is scored; "20k
  vectors are ~30 MB", and they are released when idle while keyword search keeps working.
- Keywords: FTS5 with Porter stemming over file names, folder names, OCR text, page and document text, ranked with
  `bm25(fts, 3.0, 1.0)`. A word not in the index as typed becomes a prefix query (`grok` finds `Grok4`).
- **Fusion:** cosine scores are turned into z-scores per query (and per modality, so a video frame can be ranked against a
  PDF page). A file is a candidate if its best segment has z >= 2 ("stands out"), or if it contains at least half of the
  query's words. Score = z + 2 x (word coverage)^1.5, + 1 if the query names the file's kind ("screenshot of", "in a
  video"). Author's comment: this beat reciprocal rank fusion on both their evals (R@1 0.88 vs 0.73 synthetic, 0.86 vs 0.82
  real; *author-reported*, 2026-10-07). It also gives an honest "nothing found".
- Long PDFs: first 30 pages indexed at once, the rest after everything else, up to 5,000 pages; later pages are scored
  but do not set the z-scale.
- The index records which model version made its vectors, and a test checks every llama.cpp update against stored
  reference vectors.
- No special "latest" handling: results carry `modified`, and the big window has filters.

---

## 2. Vector stores

### 2.1 What we need

| Need | Why |
|---|---|
| Fully offline, in-process, one file or folder | Mac app now, phone app next, no server |
| Works from Flutter on iOS and Android | phone app |
| Metadata filter: kind, mtime / EXIF date, folder | "latest CV", "photos from Thanksgiving", "only PDFs" |
| Keyword (BM25) next to vectors | invoice numbers, names, "CV" in a file name; no embedding holds those exactly |
| Scale: 5k to 50k vectors, 256 or 768 dims | a Downloads folder plus WhatsApp images |
| Small, permissive license | ships inside an app |

At 256 dims float32, 50,000 vectors are 51 MB; at float16, 26 MB. That fits in memory on a phone. Exact search over that
is cheap (see 2.3), so ANN indexes are optional, not required.

### 2.2 Comparison

GitHub numbers from the API on 2026-10-12. "Flutter" = a maintained way to call it from Dart on iOS and Android.

| Store | Stars | Latest release | License | Embedded | Flutter iOS/Android | Metadata filter | Keyword / hybrid | Notes |
|---|---:|---|---|---|---|---|---|---|
| **SQLite + FTS5** (plain BLOB vectors, scored in numpy / Dart / native) | n/a | SQLite 3.50.4 in Python 3.12 here | public domain | yes | yes (`package:sqlite3` 3.7.0, 2026-09-30) | SQL | **FTS5 BM25 built in** | what DigUp and Recall ship; zero extra dependency |
| [sqlite-vec](https://github.com/asg017/sqlite-vec) | 8,183 | v0.1.10-alpha.4 (2026-05-18), pip 0.1.9 | Apache-2.0 / MIT | yes (SQLite extension) | yes: [`flutter_edge_ai_sqlite`](https://pub.dev/packages/flutter_edge_ai_sqlite) 2.0.1 (2026-10-10), [`sqlite_vec`](https://pub.dev/packages/sqlite_vec) 0.1.7-alpha.3 (unverified publisher) | `vec0` metadata columns + partition keys ([release post](https://alexgarcia.xyz/blog/2024/sqlite-vec-metadata-release/index.html)) | FTS5 in the same DB, RRF in SQL ([Simon Willison](https://simonwillison.net/2024/Oct/4/hybrid-full-text-search-and-vector-search-with-sqlite/)) | exact KNN only; still pre-1.0, last push May 2026 |
| [sqlite-vector](https://github.com/sqliteai/sqlite-vector) (sqlite.ai) | 1,180 | 1.1.2 (2026-09-11) | Apache-2.0 (LICENSE file checked) | yes | yes: [`sqlite_vector`](https://pub.dev/packages/sqlite_vector) 1.1.2, verified publisher sqlite.ai | SQL on normal tables | FTS5 alongside | newer rival to sqlite-vec; quantized scan; vectors live in ordinary tables |
| [Qdrant Edge](https://qdrant.tech/edge/) | 35,049 (qdrant repo) | `qdrant_edge` 0.8.0-dev.5 (2026-10-10) | Apache-2.0 | yes, in-process (Rust) | yes: [`qdrant_edge`](https://pub.dev/packages/qdrant_edge) (repo = qdrant/qdrant, publisher not verified), [`flutter_edge_ai_qdrant`](https://pub.dev/packages/flutter_edge_ai_qdrant) | payload filters | Qdrant supports hybrid (dense + sparse); keyword on Edge not verified | HNSW; docs say **beta, API may change** |
| [ObjectBox](https://github.com/objectbox/objectbox-dart) | 1,236 (Dart) / 4,627 (Java) | Dart 5.3.2 (2026-05-20) | Apache-2.0 (bindings; core is closed binary) | yes | **yes, official**, verified publisher, 187,547 downloads / 30 days, 1,580 likes | yes (query conditions + vector) | no BM25 full-text index found in docs | HNSW since 4.0; most mature Flutter option |
| [LanceDB](https://github.com/lancedb/lancedb) | 11,651 | 0.41.0-beta.6 (2026-10-12) | Apache-2.0 | yes (Python, JS, Rust) | **no** Dart package on pub.dev; no official iOS/Android SDK found | SQL `where`, prefilter | **built-in FTS (BM25) + hybrid with RRF reranker** ([docs](https://docs.lancedb.com/search/hybrid-search/)) | best Python ergonomics for multimodal; not for the phone |
| [Chroma](https://github.com/chroma-core/chroma) (PersistentClient) | 29,485 | 1.5.9 (2026-05-05) | Apache-2.0 | yes (Python) | no | `where` metadata, `where_document` | document filter; hybrid endpoint documented for the server/cloud path, unclear locally | used by LocalLens, everythingsearch, tern |
| [DuckDB VSS](https://github.com/duckdb/duckdb-vss) | 269 | ships with DuckDB | MIT | yes | DuckDB yes ([`dart_duckdb`](https://pub.dev/packages/dart_duckdb)), VSS on mobile not verified | SQL | DuckDB FTS extension | HNSW persistence is **experimental**, no WAL recovery for the index ([docs](https://duckdb.org/docs/current/core_extensions/vss)) |
| [FAISS](https://github.com/facebookresearch/faiss) | 41,134 | v1.15.1 (2026-09-16) | MIT | yes (library) | no Flutter binding | no (do it yourself) | no | index only; overkill at 50k |
| [USearch](https://github.com/unum-cloud/USearch) | 4,333 | v2.26.4 (2026-10-05) | Apache-2.0 | yes (library, many languages) | via your own FFI / Rust bridge (what ente did); no pub.dev package | filter callback | no | tiny HNSW, f16/i8 storage; used by GPT4All and ente |
| [zvec](https://github.com/alibaba/zvec) (Alibaba, new in 2026) | 16,101 | v0.7.0 (2026-08-24) | Apache-2.0 | yes, in-process | community [`zvec`](https://pub.dev/packages/zvec) 0.5.2 (unverified publisher, 86 downloads / 30 days) | yes (per docs) | not verified | young; benchmark charts on [zvec.org](https://zvec.org/en/docs/db/benchmarks/) are vendor-run |
| Google MediaPipe `SqliteVectorStore` / `AppSearchVectorStore` | (mediapipe 37,242) | MediaPipe Tasks 1.1.0 | Apache-2.0 | yes | native iOS/Android only, needs a platform channel from Flutter | exact-match metadata map (Gallery code) | no | ships with the EG2 `SemanticRetriever`; easiest if we already call `UniversalEmbedder` natively |
| [mobile_rag_engine](https://pub.dev/packages/mobile_rag_engine) | 43 | 0.20.1 (2026-10-08) | MIT | yes (Rust core) | yes | yes | HNSW + BM25 with RRF | small project; text ONNX models only |

### 2.3 Measured on this Mac

Machine: MacBook Pro M4 Pro, Python 3.12, numpy 2.5.4, sqlite-vec 0.1.9, usearch (pip, 2026-10-12), lancedb 0.40.0.
Date: 2026-10-12. **The machine was busy (load average ~15 from other apps), so treat absolute numbers as upper bounds.**
Data: random unit vectors with a `kind` (5 values) and `mtime` column; 50 queries after 3 warm-up, top 10, median (p90).
"Filtered" = only `kind = pdf` (1 in 5 rows). Script: appendix A.

| N x dims | numpy brute force | numpy filtered | sqlite-vec `vec0` | `vec0` filtered | USearch HNSW f16 | LanceDB (no index) | LanceDB filtered |
|---|---|---|---|---|---|---|---|
| 5,000 x 256 | **0.09 ms** (0.14) | 0.07 ms | 2.4 ms (3.6) | 1.6 ms | 0.21 ms | 4.0 ms | 5.5 ms |
| 5,000 x 768 | **0.18 ms** (0.21) | 0.15 ms | 11.1 ms (21.1) | 5.7 ms | 0.63 ms | 7.4 ms | 9.6 ms |
| 50,000 x 256 | **1.4 ms** (2.1) | 1.1 ms | 25.7 ms (34.1) | 18.0 ms | 0.35 ms | 12.9 ms | 12.8 ms |
| 50,000 x 768 | **2.1 ms** (4.6) | 1.9 ms | 86.9 ms (99.4) | 48.1 ms | 1.1 ms | 24.2 ms | 30.2 ms |

On-disk size at 50,000 x 256: numpy `.npy` 51.2 MB, sqlite-vec DB 53.5 MB, USearch f16 33.0 MB, LanceDB 51.6 MB.
Insert time at 50,000 x 768: sqlite-vec 7.6 s, USearch f16 index build 25.7 s.

Correctness: sqlite-vec returned exactly the numpy top 10 in every configuration (it is exact). USearch recall@10 on these
uniform random vectors was poor (0.10 to 0.63), but uniform random high-dimensional data is the worst case for any HNSW
graph. On clustered, embedding-like data (300 centres + noise, 50,000 vectors, f16) USearch reached **recall@10 0.994 to
1.0** at 0.09 ms (256-d) and 0.23 ms (768-d) per query. Real EmbeddingGemma vectors were not used (our test set has 158).

What this says: for our sizes, **exact search is fast enough everywhere** (numpy at 2 ms, sqlite-vec under 100 ms even at
the worst case), so pick the store on filters, keywords, portability and maturity, not on ANN speed. Not measured: any
phone. Phone CPUs are slower than an M4 Pro; measure on the iPhone 16 Pro Max before choosing between `vec0` and HNSW.

### 2.4 Published numbers (not measured by us)

- **sqlite-vec vs Qdrant Edge, from Flutter's Dart VM on macOS arm64** (third party: the flutter_edge_ai maintainer,
  [benchmark doc](https://github.com/DenisovAV/flutter_edge_ai/blob/main/docs/benchmarks/rag_sqlite_vec_vs_qdrant.md),
  2026-06-21, 384 dims, vec0 v0.1.9, qdrant-edge 0.7.3): search median at 1k docs 368 us (vec0) vs 67 us (qdrant); at
  10k docs 3,422 us vs 312.5 us, top-5. Same top-K ids in every query. Insert 2,238 vs 10,719 docs/s at 10k. Author's
  summary: Qdrant Edge is 5 to 11x faster on search; the older "75x" claim was against a deleted pure-Dart store.
- **sqlite-vec full scan, 100,000 x 384 on an M1 Pro: 67.84 ms** per query
  ([Marco Bambini](https://marcobambini.substack.com/p/the-state-of-vector-search-in-sqlite), who runs sqlite.ai, the
  maker of the competing sqlite-vector, so treat as vendor-adjacent).
- **VectorDash** (author-reported): 988,658 indexed files and 34,678 vectors, semantic retrieval 64 to 238 ms on NVMe.
- **ObjectBox**: "nearest neighbours within milliseconds, even across millions" is a *vendor* claim; no published vector
  benchmark found.
- **zvec**: VectorDBBench on Cohere 1M / 10M x 768, *vendor-run*, results only as charts on
  [zvec.org](https://zvec.org/en/docs/db/benchmarks/).
- **LEANN**: "97% less storage" (60M chunks in 6 GB instead of 201 GB) is *vendor/paper* reported.

---

## 3. Recommendation

### 3.1 Mac app now: SQLite + FTS5 + numpy

One file, `<folder>/.photo-sweep/index.sqlite`, through Python's built-in `sqlite3` (FTS5 is compiled in; SQLite 3.50.4
here). No new dependency.

```
files(id, path UNIQUE, kind, ext, size, mtime, ctime, taken_at, sha1, model_rev, info JSON)
segments(id, file_id, kind, loc, text, vec BLOB)   -- vec: 256 x float32 (or 768 x float16), one row per photo / PDF page / chunk
fts USING fts5(name, body, content='segments')     -- file name + folder name + extracted text (+ OCR later)
faces(file_id, box, vec BLOB)                      -- SFace vectors, same idea
meta(key, value)                                   -- model id, image tokens (70), dims, prompt templates
```

Why:
- It replaces `cache.npz` + `cache.json` with one transactional file, keyed on path + size + mtime as now.
- "Latest", "only PDFs", "from 2024" become SQL; exact words (CV, invoice numbers, names) come from FTS5 BM25.
- Vectors are loaded once into a numpy matrix and scored by brute force: measured 1.4 ms at 50k x 256 on this Mac, and DigUp
  ships the same design.
- The file format is the one the phone will use (3.2), so an index could even be built on the Mac and copied over.
- Store `model_rev`, image tokens and dims in `meta`: 70-token and 280-token vectors are not comparable (LocalLens keeps one
  index per image-detail level; DigUp checks reference vectors on every runtime update).

What I would not pick for the Mac: LanceDB is the best Python option if we wanted built-in hybrid search and a Python-only
product, but it has no phone path, so it would mean two stores. Chroma and FAISS add weight without solving filters plus
keywords plus phone.

### 3.2 Phone app: same SQLite schema, sqlite-vec for KNN

- `package:sqlite3` (FFI) + the `vec0` extension, either through `flutter_edge_ai_sqlite` (maintained, supports all six
  platforms, metadata filters, same SQL on every platform) or by loading sqlite-vec yourself. Keep `files`, `segments`,
  `fts` identical to the Mac; add a `vec0` table with `kind` and `mtime` as metadata columns so the filtered KNN runs inside
  SQLite.
- Why not plain brute force in Dart? It would work at 5k vectors, but sqlite-vec does the scan in C with metadata filters,
  for free, and it is what Google's EG2 `SemanticRetriever` sample and flutter_edge_ai both use on iOS and Android.
- Fallback if `vec0` is too slow on the iPhone at our real size (measure it): **Qdrant Edge** (`qdrant_edge`), HNSW with
  payload filters, 5 to 11x faster than vec0 in the third-party benchmark above. Keep SQLite for files, FTS5 and faces
  either way. Risks: beta API, pub.dev publisher not verified.
- Runner-up: **ObjectBox** (most mature Flutter store, HNSW, official SDK), but it has no BM25 index, so keyword search
  would still need SQLite, and its core is a closed binary.
- Embeddings on the phone still go through a Swift / Kotlin platform channel to `UniversalEmbedder` (see
  `docs/mobile-feasibility.md`). Watch flutter_edge_ai issue #626: the maintainer said EG2 support is coming "next week";
  check whether it covers images, not only text.

### 3.3 Ideas worth copying

1. **DigUp's fusion instead of plain RRF.** Z-score the cosine scores per query (per modality if we mix photos and
   documents), keep files whose best segment has z >= 2, add files that contain at least half of the query's keywords,
   score = z + 2 x coverage^1.5, and +1 when the query names a kind ("photo", "screenshot", "PDF", "CV"). It gives an honest
   "nothing found" and handles exact names. Their evals: R@1 0.88 vs 0.73 (synthetic) and 0.86 vs 0.82 (real) against RRF,
   author-reported. Re-run on our own `rag/make_downloads.py` set before adopting the constants.
2. **Treat "latest / newest / last / this year / from 2024" as a date slot, not as meaning.** everythingsearch has an LLM
   turn the question into `q`, `source`, `date_from`, `date_to`, `match_mode`; Immich puts a time-frame filter beside smart
   search; magpie's `dl` lists the newest downloads. For us: a small rule parser (or a local typed-decision model) pulls the
   time words out, the rest ("my CV") goes to the hybrid search, then among the candidates that pass the gate we group
   near-identical document types (our test set has CVs from three years) and sort by `mtime` (or `taken_at` for photos, or a
   year found in the text). "The Thanksgiving photo" is the other direction: meaning first, with EXIF date in late November
   as a soft boost.
3. **Index PDFs per page, rank a file by its best page, and answer only from evidence.** DigUp stores a vector per PDF page
   (first 30 pages at once, the rest later) and opens the hit at that page; magpie chunks long text at ~1,600 characters
   with overlap and ranks a document by its best chunk; everythingsearch aggregates chunk scores to a file score. EG2 takes
   8,192 tokens, so a CV fits in one vector, but page-level vectors give us the page to show. For "ask": SCM runs an opt-in
   llama.cpp sidecar (Qwen3 1.7B, ~1.1 GB) that answers only from extracted evidence with numbered, clickable citations, and
   skips the model entirely when retrieval found nothing. On the phone, the same with Gemma through LiteRT-LM. (Note for
   Qwen3: turn thinking off, or it spends the whole budget thinking.)
4. Bonus: **version the index** (model, revision, image tokens, dims in `meta`) and keep a few reference vectors to detect
   a runtime change, as DigUp does; flutter_edge_ai binds each store to an "embedding profile" id for the same reason.

---

## Appendix A: benchmark script (run 2026-10-12)

Environment: `uv venv --python 3.12; uv pip install numpy sqlite-vec usearch lancedb`. Results were saved next to the
script as `bench_stores.json` and `bench_usearch_clustered.json` (scratch folder, not committed).

```python
import json, os, shutil, sqlite3, statistics, tempfile, time
import numpy as np, sqlite_vec, lancedb
from usearch.index import Index
rng = np.random.default_rng(7)

def med(fn, q, reps=50):
    for i in range(3): fn(q[i])
    ts = []
    for i in range(reps):
        t = time.perf_counter(); fn(q[i % len(q)]); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), sorted(ts)[int(0.9 * len(ts))]

for N in (5000, 50000):
    for D in (256, 768):
        X = rng.standard_normal((N, D)).astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
        Q = rng.standard_normal((60, D)).astype(np.float32); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        kind = rng.integers(0, 5, N); mtime = rng.uniform(1.6e9, 1.76e9, N); tmp = tempfile.mkdtemp()
        np_knn = lambda q: np.argpartition(-(X @ q), 10)[:10]                      # numpy brute force
        db = sqlite3.connect(os.path.join(tmp, "v.db")); db.enable_load_extension(True); sqlite_vec.load(db)
        db.execute(f"create virtual table v using vec0(emb float[{D}] distance_metric=cosine, kind integer, mtime float)")
        db.executemany("insert into v(rowid, emb, kind, mtime) values (?,?,?,?)",
                       ((i + 1, X[i].tobytes(), int(kind[i]), float(mtime[i])) for i in range(N))); db.commit()
        vec = lambda q: db.execute("select rowid, distance from v where emb match ? and k = 10 and kind = 1", (q.tobytes(),)).fetchall()
        idx = Index(ndim=D, metric="cos", dtype="f16"); idx.add(np.arange(N), X)   # USearch HNSW
        tbl = lancedb.connect(os.path.join(tmp, "lance")).create_table("t", data=[
            {"vector": X[i].tolist(), "kind": int(kind[i]), "mtime": float(mtime[i])} for i in range(N)])
        print(N, D, med(np_knn, Q), med(vec, Q), med(lambda q: idx.search(q, 10), Q),
              med(lambda q: tbl.search(q).metric("cosine").where("kind = 1", prefilter=True).limit(10).to_list(), Q, 30))
        shutil.rmtree(tmp)
```

The clustered-data recall check used 300 Gaussian centres, `X = centre + 0.6 * noise`, 50,000 vectors, USearch f16 with
`expansion_search` 64 and 256, recall@10 against numpy exact top 10 over 50 queries.
