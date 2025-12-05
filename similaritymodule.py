import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from skimage.metrics import structural_similarity as ssim

# ============================================
# 1) 딥러닝 임베딩 모델 로드
# ============================================

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_7.h5")

_EMBEDDER = None
_HWC = None

def _load_embedder():
    global _EMBEDDER, _HWC
    if _EMBEDDER is not None:
        return

    # 모델 파일이 없으면 더미 모드로 동작 (에러 방지)
    if not os.path.exists(MODEL_PATH):
        print(f"[SIMILARITY] Model not found at {MODEL_PATH}")
        return

    try:
        base = load_model(MODEL_PATH, compile=False)
        in_shape = tuple(base.input_shape[1:])
        _HWC = in_shape

        inp = tf.keras.Input(shape=in_shape)
        x = inp
        for layer in base.layers[:-1]:
            x = layer(x)
        _EMBEDDER = Model(inp, x, name="embedder")
        print(f"[SIMILARITY] Embedder loaded. Input={in_shape}")
    except Exception as e:
        print(f"[SIMILARITY] Failed to load model: {e}")
        _EMBEDDER = None
        _HWC = None

# ============================================
# 2) 이미지 전처리 및 측정 함수들
# ============================================

def _preprocess_dl(img, size):
    """딥러닝 모델용 전처리"""
    # Grayscale -> RGB (모델이 3채널일 경우)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img_resized = cv2.resize(img, (size[1], size[0]))
    x = (img_resized.astype("float32") / 255.0)[None, ...]
    return x

def _get_embedding(img):
    """딥러닝 임베딩 추출"""
    _load_embedder()
    if _EMBEDDER is None or _HWC is None:
        # 모델이 없으면 그냥 128-dim zero 벡터 반환 (fallback)
        return np.zeros(128, dtype="float32")

    H, W, C = _HWC
    x = _preprocess_dl(img, size=(H, W))
    e = _EMBEDDER.predict(x, verbose=0)[0]
    return e / (np.linalg.norm(e) + 1e-9)

def _calc_ssim(img1, img2):
    """3. 구조적 정확도 (SSIM)"""
    # 흑백 변환 및 크기 통일
    i1 = cv2.resize(img1, (128, 128))
    i2 = cv2.resize(img2, (128, 128))
    if i1.ndim == 3:
        i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    if i2.ndim == 3:
        i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

    score, _ = ssim(i1, i2, full=True)
    return float(max(0.0, min(1.0, score)))

def _calc_hist_corr(img1, img2):
    """4. 획 두께/농도 (히스토그램 상관관계)"""
    # 흑백 변환
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 히스토그램 계산
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

    # 정규화
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # 상관관계 비교 (1.0에 가까울수록 분포가 비슷함)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, score)))

def _calc_shape_sim(img1, img2):
    """5. 글자 외형 (Hu Moments)"""
    # 이진화 (Thresholding)
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    _, th1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY_INV)
    _, th2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY_INV)

    # Hu Moments 기반 shape distance (낮을수록 비슷)
    d = cv2.matchShapes(th1, th2, cv2.CONTOURS_MATCH_I1, 0)

    # 점수화: 거리가 0이면 1점, 멀어질수록 0으로 수렴
    score = 1.0 / (1.0 + d)
    return float(max(0.0, min(1.0, score)))

# ============================================
# 3) 메인 함수
# ============================================

TARGET_TEXT = "동해물과백두산이마르고닳도록"

def compute_similarity(job_id: str, out_dir: str, handwriting_subdir: str = "handwriting") -> dict:
    """
    job_id: "1764..." 같은 문자열
    out_dir: result{jobId} 디렉토리 경로 (app.py에서 넘겨줌)
    handwriting_subdir: "handwriting" 또는 "retry"
    """
    _load_embedder()

    job_root = out_dir
    hand_dir = os.path.join(job_root, handwriting_subdir)
    gen_dir  = os.path.join(job_root, "generation")

    if not os.path.isdir(hand_dir) or not os.path.isdir(gen_dir):
        print(f"[SIMILARITY] missing dir: hand={hand_dir}, gen={gen_dir}")
        return {
            "AI 필체 유사도": 0.0,
            "특징 일치도": 0.0,
            "구조적 정확도": 0.0,
            "획 농도": 0.0,
            "글자 외형": 0.0,
        }

    # 매칭되는 파일 쌍 찾기
    pairs = []

    # (주의) 중간 발표용으로 1글자만 생성했다면, 생성된 파일이 있는 것만 비교합니다.
    for idx, ch in enumerate(TARGET_TEXT, start=1):
        # 생성본 이름: jobId_generated_c{idx}.png
        gen_name = f"{job_id}_generated_c{idx}.png"
        g_path = os.path.join(gen_dir, gen_name)
        if not os.path.exists(g_path):
            continue

        # 손글씨 쪽은 다양한 네이밍 가능성(유니코드, 0.png 등)을 모두 케어
        unicode_name = f"{ord(ch)}.png"
        numeric_name = f"{idx-1}.png"   # 0.png,1.png,... 패턴

        candidates = [
            os.path.join(hand_dir, unicode_name),
            os.path.join(hand_dir, numeric_name),
        ]

        hand_path = None
        for cand in candidates:
            if os.path.exists(cand):
                hand_path = cand
                break

        if hand_path is None:
            print(f"[SIMILARITY] no handwriting file for char '{ch}' (idx={idx})")
            continue

        pairs.append((hand_path, g_path))

    print(f"[SIMILARITY] matched pairs = {len(pairs)}")

    if not pairs:
        print("[SIMILARITY] No pairs found.")
        return {
            "AI 필체 유사도": 0.0,
            "특징 일치도": 0.0,
            "구조적 정확도": 0.0,
            "획 농도": 0.0,
            "글자 외형": 0.0,
        }

    # 지표별 점수 리스트
    scores_cos = []
    scores_l2 = []
    scores_ssim = []
    scores_hist = []
    scores_shape = []

    for h_path, g_path in pairs:
        # 이미지 로드
        img_h = cv2.imread(h_path, cv2.IMREAD_UNCHANGED) # 원본
        img_g = cv2.imread(g_path, cv2.IMREAD_UNCHANGED) # 생성본
        if img_h is None or img_g is None:
            print(f"[SIMILARITY] failed to read: {h_path} or {g_path}")
            continue

        # 알파 채널 제거 / 채널 통일
        if img_h.ndim == 3 and img_h.shape[-1] == 4:
            img_h = cv2.cvtColor(img_h, cv2.COLOR_BGRA2BGR)
        if img_g.ndim == 3 and img_g.shape[-1] == 4:
            img_g = cv2.cvtColor(img_g, cv2.COLOR_BGRA2BGR)
        if img_h.ndim == 2:
            img_h = cv2.cvtColor(img_h, cv2.COLOR_GRAY2BGR)
        if img_g.ndim == 2:
            img_g = cv2.cvtColor(img_g, cv2.COLOR_GRAY2BGR)

        # 1. AI Cosine
        emb_h = _get_embedding(img_h)
        emb_g = _get_embedding(img_g)
        cos_sim = float(np.dot(emb_h, emb_g))
        scores_cos.append(cos_sim)

        # 2. AI L2 (거리를 점수화: 0에 가까우면 1점)
        l2_dist = float(np.linalg.norm(emb_h - emb_g))
        l2_score = 1.0 / (1.0 + l2_dist)
        scores_l2.append(l2_score)

        # 3. SSIM
        scores_ssim.append(_calc_ssim(img_h, img_g))

        # 4. Histogram
        scores_hist.append(_calc_hist_corr(img_h, img_g))

        # 5. Shape (Hu Moments)
        scores_shape.append(_calc_shape_sim(img_h, img_g))

    if not scores_cos:
        # 로딩 실패 등으로 아무 것도 못 계산한 경우
        return {
            "AI 필체 유사도": 0.0,
            "특징 일치도": 0.0,
            "구조적 정확도": 0.0,
            "획 농도": 0.0,
            "글자 외형": 0.0,
        }

    cos_mean   = float(np.mean(scores_cos))
    l2_mean    = float(np.mean(scores_l2))
    ssim_mean  = float(np.mean(scores_ssim))
    hist_mean  = float(np.mean(scores_hist))
    shape_mean = float(np.mean(scores_shape))

    # ✅ 프론트가 기대하는 키 기준(레이더 차트용)
    metrics = {
        "cosine similarity":    round(cos_mean,  2),  # 1. cos sim
        "L2 Distance":          round(l2_mean,   2),  # 2. L2 score
        "SSIM(구조적 정확도)":  round(ssim_mean, 2),  # 3. SSIM
        "획 두께 농도":         round(hist_mean, 2),  # 4. histogram corr
        "글자 외형":           round(shape_mean,2),  # 5. shape
    }

    # 🔹 한글/옛날 키도 같이 내려주고 싶으면 alias 로 추가
    metrics.update({
        "AI 필체 유사도": metrics["cosine similarity"],
        "특징 일치도":   metrics["L2 Distance"],
        "구조적 정확도": metrics["SSIM(구조적 정확도)"],
        "획 농도":       metrics["획 두께 농도"],
    })

    # (예전 legacy 키 유지하고 싶으면 필요에 따라 추가)
    # legacy = {
    #     "균형":   metrics["SSIM(구조적 정확도)"],
    #     "획간격": metrics["cosine similarity"],
    #     "기울기": metrics["글자 외형"],
    #     "자간":   metrics["L2 Distance"],
    #     "획두께": metrics["획 두께 농도"],
    # }
    # metrics.update(legacy)

    print(f"[SIMILARITY METRICS] {metrics}")
    return metrics
