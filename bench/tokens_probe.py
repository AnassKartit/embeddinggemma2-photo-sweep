import time, torch, sys
from sentence_transformers import SentenceTransformer
from PIL import Image
m = SentenceTransformer("google/embeddinggemma-2", device="mps", config_kwargs={"audio_config": None}, model_kwargs={"torch_dtype": torch.float32})
proc = m[0].processor
img = Image.open(sorted(__import__("glob").glob("data/other_people/*.jpg"))[0]).convert("RGB")
ref = None
for tok in (280, 140, 70):
    proc.image_processor.max_soft_tokens = tok; proc.image_processor.image_seq_length = tok; proc.image_seq_length = tok
    m.encode([{"image": img}] * 8, normalize_embeddings=True)
    t = time.time(); e = m.encode([{"image": img}] * 16, batch_size=8, normalize_embeddings=True); dt = (time.time() - t) / 16
    ref = e[0] if ref is None else ref
    print(tok, "ms/img", round(dt * 1000, 1), "cos vs 280:", round(float(e[0] @ ref), 4))
