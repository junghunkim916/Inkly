import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from skimage.metrics import structural_similarity as ssim

# ============================================
# 설정
# ============================================

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_7.h5")

_EMBEDDER = None
_HWC = None
_EMB_DIM = None
_FALLBACK = False


# ============================================
# 임베딩 로드
# ============================================

def _load_embedder():
    global _EMBEDDER, _HWC, _EMB_DIM, _FALLBACK
    if _EMBEDDER is not None or _FALLBACK:
        return

    if not os.path.exists(MODEL_PATH):
        print("[SIM] model not found → fallback")
        _FALLBACK = True
        _EMB_DIM = 128
        return

    base = load_model(MODEL_PATH, compile=False)
    _HWC = tuple(base.input_shape[1:])

    inp = tf.keras.Input(shape=_HWC)
    x = inp
    for layer in base.layers[:-1]:
        x = layer(x)
    _EMBEDDER = Model(inp, x)

    dummy = np.zeros((1, *_HWC), dtype="float32")
    _EMB_DIM = int(_EMBEDDER.predict(dummy, verbose=0).shape[-1])

    print(f"[SIM] embedder loaded: {_HWC}, dim={_EMB_DIM}")


# ============================================
# 유틸
# ============================================

def _to_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _fallback_embedding(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (64, 64))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    v = np.concatenate([gx.flatten(), gy.flatten()])[:128]
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _get_embedding(img):
    _load_embedder()
    img = _to_bgr(img)
    if img is None:
        return np.zeros(_EMB_DIM or 128, dtype="float32")

    if _FALLBACK or _EMBEDDER is None:
        return _fallback_embedding(img)

    H, W, _ = _HWC
    x = cv2.resize(img, (W, H))
    x = (x.astype("float32") / 255.0)[None, ...]
    e = _EMBEDDER.predict(x, verbose=0)[0]
    return e / (np.linalg.norm(e) + 1e-9)


def _calc_ssim(img1, img2):
    g1 = cv2.cvtColor(cv2.resize(img1, (128,128)), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(img2, (128,128)), cv2.COLOR_BGR2GRAY)
    score, _ = ssim(g1, g2, full=True)
    return float(score)


def _calc_hist(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h1 = cv2.calcHist([g1],[0],None,[256],[0,256])
    h2 = cv2.calcHist([g2],[0],None,[256],[0,256])
    cv2.normalize(h1,h1)
    cv2.normalize(h2,h2)

    corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    # [-1, 1] → [0, 1]
    return float((corr + 1.0) / 2.0)


def _calc_shape(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, t1 = cv2.threshold(g1, 200, 255, cv2.THRESH_BINARY_INV)
    _, t2 = cv2.threshold(g2, 180, 255, cv2.THRESH_BINARY_INV)
    if cv2.countNonZero(t1) < 10 or cv2.countNonZero(t2) < 10:
        return 0.0
    d = cv2.matchShapes(t1, t2, cv2.CONTOURS_MATCH_I1, 0)
    return float(1.0 / (1.0 + d))


# ============================================
# 핵심: 0.png 하나만 비교
# ============================================

def compute_similarity(job_id: str, out_dir: str, handwriting_subdir: str = "handwriting") -> dict:
    hand_path = os.path.join(out_dir, handwriting_subdir, "0.png")
    gen_path  = os.path.join(out_dir, "generation", "0.png")

    if not os.path.exists(hand_path) or not os.path.exists(gen_path):
        print("[SIM] 0.png not found")
        return _empty()

    img_h = _to_bgr(cv2.imread(hand_path, cv2.IMREAD_UNCHANGED))
    img_g = _to_bgr(cv2.imread(gen_path,  cv2.IMREAD_UNCHANGED))

    emb_h = _get_embedding(img_h)
    emb_g = _get_embedding(img_g)

    return {
        "AI 필체 유사도": round(float(np.dot(emb_h, emb_g)), 2),
        "특징 일치도": round(1.0 / (1.0 + float(np.linalg.norm(emb_h - emb_g))), 2),
        "구조적 정확도": round(_calc_ssim(img_h, img_g), 2),
        "획 농도": round(_calc_hist(img_h, img_g), 2),
        "글자 외형": round(_calc_shape(img_h, img_g), 2),
    }


def _empty():
    return {
        "AI 필체 유사도": 0.0,
        "특징 일치도": 0.0,
        "구조적 정확도": 0.0,
        "획 농도": 0.0,
        "글자 외형": 0.0,
    }
