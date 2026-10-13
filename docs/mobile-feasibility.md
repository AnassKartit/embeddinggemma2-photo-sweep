# Mobile feasibility: Flutter photo-cleanup demo with EmbeddingGemma 2 + faces on device

Research date: 12 October 2026. Nothing was installed or built. Every number below is either read from a
file listing / source file (linked) or quoted from a vendor page and labelled "vendor". Nothing here was measured on a phone yet.

## Verdict

Feasible this week on the iPhone 16 Pro Max, with one caveat: **no Flutter plugin does EmbeddingGemma 2 image
embeddings today**. The clean path is a small Swift (iOS) / Kotlin (Android) platform channel around Google's
official `UniversalEmbedder` (MediaPipe Tasks 1.1.0) or `EmbeddingEngine` (LiteRT-LM 0.18.0). Faces port 1:1 from
the Mac pipeline with `opencv_dart` (same `yunet.onnx` + `sface.onnx`, same OpenCV API, so the 0.45 / 2-face rule carries over).

## 1. On-device artifacts (Hugging Face, litert-community, all Apache-2.0, not gated)

Sizes are exact bytes from the HF API tree listing.

| Repo | Modalities | Generic CPU/GPU file | Size |
|---|---|---|---|
| [litert-community/embeddinggemma-2-text-vision-440m-litert-lm](https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm) | text + image | `embeddinggemma-2-text-vision-440m.litertlm` | 387,710,976 B (388 MB) |
| [litert-community/embeddinggemma-2-740m-litert-lm](https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm) | text + image + video + audio | `embeddinggemma-2-740m.litertlm` | 484,622,336 B (485 MB) |
| [litert-community/embeddinggemma-2-text-270m-litert-lm](https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm) | text only (no vision) | `embeddinggemma-2-text-270m.litertlm` | 164,626,432 B (165 MB) |

- **Vision encoder is included** in the 440M and 740M files (170M-param encoder). For photo cleanup, **use the 440M text-vision file**: smaller, and it is the one Google's iOS/Android samples load.
- Quantization (740M card): transformer and embeddings **int4** per-channel QAT; vision encoder **int8** per-channel QAT; audio mixed int2/4/8.
- Image soft-token signatures: **70 and 140** (our Mac default of 70 is supported). Output sizes 128/256/512/768 (Matryoshka). Text signatures 128 to 8192.
- No `.tflite`; the format is `.litertlm` only. Each repo also has NPU-specific files (Tensor G5/G6, Qualcomm SM8550/8650/8750/8850/QCS8275, MediaTek MT6991/6993, Intel PTL), 447 to 898 MB, for Android NPU only. iPhone uses the generic file.
- Vendor latency, text+vision at 70 tokens (from the 740M card): iPhone 18 Pro GPU 69.8 ms / CPU 191 ms; iPhone memory 196 MB (vendor). S26 Ultra GPU 119 ms; Pixel 11 Pro TPU 49 ms. **No iPhone 16 number exists**; measure it.
- Other formats exist (fallbacks): [onnx-community/embeddinggemma-2-ONNX](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX) (`vision_encoder_q4.onnx_data` 108,957,696 B, `model_q4.onnx_data` 174,028,800 B, needs your own pre/post-processing), [FluidInference/embeddinggemma-2-coreml](https://huggingface.co/FluidInference/embeddinggemma-2-coreml) (`EmbeddingGemma2Vision.mlpackage` weights 304,078,400 B), plus MLX and GGUF ports.

## 2. Calling image embedding

### (a) Android

Option A, MediaPipe Tasks (what the Gallery app uses):
- Gradle: `com.google.mediapipe:tasks-retrieval:1.1.0` (only version on Google Maven; it pulls LiteRT-LM). Gallery also adds `com.google.ai.edge.litertlm:litertlm-android:0.18.0` and uses `minSdk = 31`.
- API (package `com.google.mediapipe.tasks.retrieval.universalembedder`): `UniversalEmbedderOptions.builder().setBaseOptions(BaseOptions.builder().setModelAssetPath(path).setDelegate(Delegate.GPU).build()).setL2Normalize(true).setVisionTokensPerImage(70).setVisionDelegate(..).build()`, then `UniversalEmbedder.createFromOptions(context, options)`; `embedImage(MPImage)`, `embedText(String)`, `embedContent(List<Any>)` (a `ByteArray` is treated as an encoded image), `UniversalEmbedder.cosineSimilarity(..)`, `close()`. Result: `result.embeddings()[0].floatEmbedding()`.
- Docs: <https://developers.google.com/edge/mediapipe/solutions/retrieval/universal_embedder/android>, sample: <https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/universal_embedder/android>.

Option B, LiteRT-LM directly: `com.google.ai.edge.litertlm:litertlm-android:0.18.0`, class `com.google.ai.edge.litertlm.EmbeddingEngine(EmbeddingEngineConfig(modelPath=..))`, `initialize()`, `computeEmbedding(listOf(InputData...))`, `computeEmbeddingAsync`, `computeEmbeddingBatch`. Source: [kotlin/.../EmbeddingEngine.kt @ v0.18.0](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.18.0/kotlin/java/com/google/ai/edge/litertlm/EmbeddingEngine.kt).

ML Kit GenAI for EmbeddingGemma 2: announced "in the coming weeks" (<https://developers.google.com/ml-kit/genai>); not available yet. Not needed.

### (b) iOS

Option A, MediaPipe Tasks:
- **Swift Package Manager only**: `.package(url: "https://github.com/google-ai-edge/mediapipe.git", from: "1.1.0")`, product `MediaPipeTasksRetrieval`, `import MediaPipeTasksRetrieval`. Package.swift declares iOS 15+. Binary xcframeworks from `dl.google.com/cpdc/20261005-190020/...-1.1.0.xcframework.zip`. **No CocoaPod**: trunk has `MediaPipeTasksVision`/`Common` up to 1.0.0 but no `MediaPipeTasksRetrieval`.
- API: `let o = UniversalEmbedderOptions(); o.baseOptions.modelAssetPath = path; o.visionDelegate = .GPU; o.l2Normalize = true; o.visionTokensPerImage = 70` then `UniversalEmbedder(options: o)`; `embed(image: MPImage(uiImage:))`, `embed(imageBytes: Data)`, `embed(text:)`, `embed(content: [Any])`, `UniversalEmbedder.cosineSimilarity(embedding1:embedding2:)`. Vector: `result.embeddings[0].floatEmbedding`. Options header confirms `visionTokensPerImage`, `maxInputLength`, `activationDataType`, `cacheDirectory`, per-modality `textDelegate/visionDelegate/audioDelegate` ([MPPUniversalEmbedderOptions.h](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/ios/retrieval/universal_embedder/sources/MPPUniversalEmbedderOptions.h)).
- Official sample (Xcode project, SPM, downloads the 440M file via `RunScripts/download_models.sh`, notes simulator falls back to CPU): <https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/universal_embedder/ios>. Docs: <https://developers.google.com/edge/mediapipe/solutions/retrieval/universal_embedder/ios>.

Option B, LiteRT-LM Swift: SPM `https://github.com/google-ai-edge/LiteRT-LM` tag `v0.18.0`, product `LiteRTLM` (iOS 15+, macOS 12+). `actor EmbeddingEngine(config: EmbeddingEngineConfig(modelPath:, backend:, visionBackend:, cacheDir:, maxInputLength:, visionTokensPerImage:))`, `initialize()`, `computeEmbedding(contents: [.imageData(Data)] / [.imageFile(path)] / [.text(..)], options: EmbeddingOptions(normalize:, outputSize: 256, visionTokensPerImage: 70))` returns `EmbeddingResponse.embedding: [Float]`. `outputSize: 256` gives the Matryoshka cut natively. A C API (`c/embedding_engine.h`, `litert_lm_embedding_engine_compute_embedding`) also exists, so a pure `dart:ffi` binding is possible later.

### (c) Flutter

No plugin supports EmbeddingGemma 2 image embeddings today:

| Package | Latest | EG2 images? |
|---|---|---|
| `flutter_gemma` | 1.11.4 (5 Oct 2026), final release, renamed | no |
| `flutter_edge_ai` (+ `_embeddings`, `_mediapipe`, `_litertlm`) | 2.2.1 / 2.2.3 (12 Oct 2026), MIT | no: only EmbeddingGemma v1 300M text. [Issue #626](https://github.com/DenisovAV/flutter_edge_ai/issues) "support EmbeddingGemma 2", maintainer: "plan to release it next week" |
| `litertlm` (yangyuan) | 0.0.14, LiteRT-LM 0.17.0 | no: chat `Engine`/`Conversation` only, no embedding API |
| `flutter_litert_lm` | 0.3.0 (Apr 2026) | no: LLM chat |
| `flutter_litert` / `tflite_flutter` | 3.9.4 / 0.12.1 | `.tflite` only; cannot load `.litertlm` |
| `mediapipe_core/_text/_genai` | 0.0.1 (2024) | abandoned, no retrieval |

**Cleanest path: a platform channel (or Pigeon) in the app itself**, not a plugin: Swift `UniversalEmbedder` in the iOS Runner (MediaPipe added as an SPM package in `Runner.xcodeproj`, which coexists with CocoaPods), Kotlin `UniversalEmbedder` in Android `MainActivity`. Channel surface: `init(modelPath, tokens)`, `embedImage(path) -> Float32List`, `embedText(String) -> Float32List`. Run it off the main thread (MediaPipe calls are synchronous). Re-check `flutter_edge_ai` next week; if EG2 lands, swap.

## 3. Google AI Edge Gallery, Instant Media Search

Cloned <https://github.com/google-ai-edge/gallery> at `b9c4594` (12 Oct 2026). License **Apache-2.0**: reusable with attribution and NOTICE.
- In code the feature is `customtasks/smartalbum/` ("Instant Media Search" in `strings.xml`).
- Runtime on Android: **MediaPipe `UniversalEmbedder`** (`services/semanticretrieval/OnDeviceEmbedder.kt`, class `UniversalOnDeviceEmbedder`) with `SemanticRetriever`; `tasks-retrieval 1.1.0`, `litertlm-android 0.18.0`, `minSdk 31`. `DEFAULT_VISION_TOKEN_BUDGET = 70`, `setL2Normalize(true)`, GPU first with automatic CPU fallback on init or inference failure, per-config cache dir (`cache_v70_t256`), passes images as encoded `ByteArray` to `embedContent`, reads limits with `ModelInfo.from(path).maxVisionTokenBudget()`.
- Model allowlist `model_allowlists/1_0_20.json`: `litert-community/embeddinggemma-2-740m-litert-lm`, file `embeddinggemma-2-740m.litertlm`, 484,622,336 B, accelerators `gpu,cpu`.
- **iOS source is not in the repo** (Android sources only, plus `ios_1_0_0.json` allowlist with no EmbeddingGemma entry). The iOS runtime cannot be confirmed from source; the closest public iOS code is the MediaPipe sample above.
- Worth copying for Android: `OnDeviceEmbedder.kt` (GPU to CPU fallback, model-limit capping) and the indexing worker pattern.

## 4. Faces on mobile

Recommended: **`opencv_dart` 2.2.2** (Apache-2.0, dart:ffi + native-assets hooks, needs Flutter >= 3.38; iOS arm64 and Android arm64/armv7/x86_64). It binds `FaceDetectorYN.fromFile(...)`, `.detect(Mat)`, and `FaceRecognizerSF.fromFile(model, config)`, `.alignCrop`, `.feature`, `.match(.., disType: FR_COSINE)`. That is the exact API `bench/faces.py` uses, so `models/yunet.onnx` (233 KB) and `models/sface.onnx` (39 MB) and the thresholds move unchanged. Mac CPU: 19 ms/photo (REVIEW.md); phone speed unmeasured.

Alternatives:
- `flutter_onnxruntime` 1.9.0 (Oct 2026) for both ONNX files: you must rewrite YuNet prior decoding/NMS and SFace 5-point alignment in Dart. More work, smaller binary.
- `google_mlkit_face_detection` 0.15.1 (detect + landmarks) or Apple Vision (iOS only), then SFace via ONNX Runtime with landmark-based alignment. Detector differs from the Mac, so re-tune.
- `face_detection_tflite` 6.10.1 has its own detection + embedding models; different embedding space, re-tune everything.

## 5. Local tooling (checked, nothing installed)

| Item | State |
|---|---|
| Flutter | `/opt/homebrew/bin/flutter`, 3.47.1 stable, Dart 3.13.1 (an upgrade is offered). `enable-swift-package-manager` not set |
| Xcode | 26.5 (17F42), iOS SDK 26.5; CocoaPods 1.16.2 |
| iOS DDI | `/Library/Developer/DeveloperDiskImages/iOS_DDI`, build 27A5194q, `contentIsCompatible: true` |
| iPhone | "AK's iPhone 16" = iPhone 16 Pro Max (iPhone17,2), iOS **27.0.1**, paired, Developer Mode enabled, available |
| Android SDK | `~/Library/Android/sdk`: platforms 31, 33-36; build-tools 30.0.3 to 36.1.0; cmake 3.22.1; emulator; **no NDK** (Gradle fetches it when a plugin needs it) |
| JDK | OpenJDK 21.0.12.1 |
| Android device | `adb devices`: **none connected** |

## Recommended architecture

```
Flutter UI (piles: Keep / Forwards / Screenshots / Receipts / Accidental / Duplicates, search box)
  photo_manager 3.12.0 -> thumbnails (long side ~1024) -> temp JPEG
  |-- MethodChannel "embed" -> iOS Swift UniversalEmbedder (SPM MediaPipeTasksRetrieval 1.1.0, GPU, 70 tokens)
  |                         -> Android Kotlin UniversalEmbedder (tasks-retrieval 1.1.0)
  |   -> Dart: slice to 256 dims, renormalise, store int8 in sqlite (sqflite)
  |-- opencv_dart: YuNet + SFace (bundled ONNX), 128-d face vectors in sqlite
  `-- Dart: zero-shot labels (text embeddings), cosine dup check, k-NN family faces, blur/dark check
Model: embeddinggemma-2-text-vision-440m.litertlm (388 MB) downloaded once to app documents (or pushed with devicectl for the demo); not a Flutter asset.
```

## Confirmed vs unconfirmed

Confirmed (from files/source): artifact names and sizes; 70-token signature; MediaPipe 1.1.0 SPM product and API on iOS and Android; LiteRT-LM 0.18.0 `EmbeddingEngine` in Swift/Kotlin/C; Gallery uses `UniversalEmbedder` on Android with 70 tokens; Apache-2.0; no Flutter plugin with EG2 images; opencv_dart exposes YuNet/SFace; toolchain and iPhone ready.

Unconfirmed: iPhone 16 Pro Max latency and memory (vendor numbers are for iPhone 18 Pro); whether int4/int8 `.litertlm` vectors match the Mac sentence-transformers vectors closely enough to keep the 0.93 duplicate cut and zero-shot labels (check first on the Mac with `pip install litert-lm` `EmbeddingEngine` on the 158-image set); Gallery's iOS runtime; MediaPipe and opencv_dart linking cleanly together in one iOS binary; Android behaviour on a real device (none connected).

## Effort (one developer)

- Day 0.5: Mac parity check, LiteRT-LM Python vs sentence-transformers on the 158 images, recalibrate thresholds.
- Days 1-2: Flutter app + iOS Swift channel + opencv_dart faces, running on the iPhone with a 200-photo library.
- Day 3: piles UI, search, timing/memory readout on device.
- Days 4-5: Android channel (copy Gallery's embedder) once a device is connected.

iPhone demo this week: realistic. Both platforms polished: about one week.

## Risks

1. Embedding drift from QAT int4/int8 vs the Mac model: thresholds may move. Mitigation: the Mac parity check above.
2. Model size: 388 MB download, ~200 MB RAM (vendor); background indexing of a large library needs batching, and iOS may suspend it.
3. iOS packaging: MediaPipe Retrieval is SPM-only while Flutter plugins default to CocoaPods; add it to the Runner by hand. opencv_dart adds a large native binary; possible duplicate-symbol clashes with MediaPipe (untested).
4. Toolchain: building with the iOS 26.5 SDK for an iOS 27.0.1 phone. The DDI reports compatible, so it should work, but it has not been tried with this project.
5. Face thresholds came from synthetic Codex families (REVIEW.md caveat); re-check on real photos with consent.
6. Plugin churn: `flutter_edge_ai` may ship EG2 next week, which would make the hand-written channel optional.
