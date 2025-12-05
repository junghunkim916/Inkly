from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageDraw
import os, io, time, random
import mathplotlib

from parsingmodule import parse_to_char_images
from generatemodule import generate_from_chars
from similaritymodule import compute_similarity

from threading import Thread
from generatemodule import generate_from_chars, TARGET_TEXT, preprocess_char_pil, IMG_SIZE

JOB_STATE = {}  # { jobId: {"state": "pending|running|done|error", "rep": "...", "error": "..."} }

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)
RESULT_DIR = BASE_DIR  # result{jobId} 들이 있는 루트 디렉토리

# app.py 상단 근처
import os, time
def _run_generate_job(job_id: str):
    """실제 무거운 보간 작업을 백그라운드에서 수행"""
    try:
        job_root = os.path.join(BASE_DIR, f"result{job_id}")
        handwriting_dir = os.path.join(job_root, "handwriting")
        if not os.path.isdir(handwriting_dir):
            JOB_STATE[job_id] = {"state": "error", "error": f"no handwriting for job {job_id}"}
            return

        # handwriting 안의 PNG들
        all_pngs = [f for f in os.listdir(handwriting_dir) if f.lower().endswith(".png")]
        if not all_pngs:
            JOB_STATE[job_id] = {"state": "error", "error": "no handwriting images"}
            return

        def _nat_key(name: str):
            base = os.path.splitext(os.path.basename(name))[0]
            try:
                return int(base)
            except ValueError:
                return 10_000

        all_pngs.sort(key=_nat_key)

        from generatemodule import TARGET_TEXT

        # 🔴 중간 시연용: "동" 하나만 생성
        dong_unicode = f"{ord(TARGET_TEXT[0])}.png"
        if dong_unicode in all_pngs:
            first_char_file = dong_unicode
        elif "0.png" in all_pngs:
            first_char_file = "0.png"
        else:
            first_char_file = all_pngs[0]

        char_files = [first_char_file]

        out = generate_from_chars(
            char_files=char_files,
            out_dir=handwriting_dir,
            job_id=job_id,
            size=None
        )

        rep_name = out["representative"]               # 예: f"{job_id}_compare.png"
        rep_rel  = f"result{job_id}/generation/{rep_name}"
        rep_abs  = os.path.join(BASE_DIR, rep_rel)

        # ✅ 여기서 진짜 파일이 생겨서 안정될 때까지 기다렸다가 done 으로 바꿈
        ok = wait_for_file(rep_abs, timeout=30.0, poll=0.2, require_stable=True)
        if not ok:
            JOB_STATE[job_id] = {
                "state": "error",
                "error": f"representative not ready: {rep_rel}"
            }
            return

        JOB_STATE[job_id] = {
            "state": "done",
            "rep": rep_rel,
        }

    except Exception as e:
        JOB_STATE[job_id] = {"state": "error", "error": str(e)}
def wait_for_file(path: str, timeout: float = 15.0, poll: float = 0.2, require_stable: bool = True) -> bool:
    """
    path가 생성될 때까지 timeout 동안 폴링한다.
    require_stable=True면, 파일 사이즈가 두 번 연속 동일해야 '안정'으로 간주.
    반환: True(발견/안정), False(타임아웃)
    """
    deadline = time.time() + timeout
    last_size = None
    stable_count = 0

    while time.time() < deadline:
        if os.path.exists(path):
            if not require_stable:
                return True
            try:
                size = os.path.getsize(path)
            except OSError:
                size = None
            if size is not None and size > 0:
                if size == last_size:
                    stable_count += 1
                    if stable_count >= 2:  # 두 번 연속 동일하면 안정
                        return True
                else:
                    stable_count = 0
                    last_size = size
        time.sleep(poll)
    return False

@app.route("/healthz")
def healthz():
    return jsonify(ok=True, ts=time.time())

# ① 업로드: 단순히 받은 파일을 저장만
BASE_DIR = os.path.dirname(__file__)

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify(ok=False, error="no file"), 400

    job_id_from_fe = request.form.get("jobId")
    job_id = job_id_from_fe or str(int(time.time()))

    job_root = os.path.join(BASE_DIR, f"result{job_id}")
    os.makedirs(job_root, exist_ok=True)

    # 원본은 grid_{jobId}.png로 저장
    save_path = os.path.join(job_root, f"grid_{job_id}.png")
    f.save(save_path)
    print("[UPLOAD]", save_path)

    try:
        char_files = parse_to_char_images(
            src_path=save_path,
            out_dir=None,   # 무시됨
            count=14,
            prefix=job_id,
        )
        print("[PARSING] chars:", char_files)
    except Exception as e:
        return jsonify(
            ok=True,
            filename=f"{job_id}_hand.png",
            parsed=[],
            parse_error=str(e),
            jobId=job_id,
        )

    return jsonify(
        ok=True,
        filename=f"{job_id}_hand.png",
        parsed=char_files,
        jobId=job_id,
    )

# ② 글씨체 생성(모의)
import os
import time
from flask import jsonify, request

BASE_DIR = os.path.dirname(__file__)

# ② 글씨체 생성
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    raw_jid = data.get("jobId")
    if raw_jid is None:
        return jsonify(ok=False, error="jobId is required"), 400

    jobId = str(raw_jid).strip()
    if not jobId:
        return jsonify(ok=False, error="invalid jobId"), 400

    # 이미 돌고 있는지 체크
    st = JOB_STATE.get(jobId)
    if st and st.get("state") in ("running", "done"):
        # 이미 실행 중이거나 끝난 job
        return jsonify(ok=True, jobId=jobId, state=st["state"])

    JOB_STATE[jobId] = {"state": "running"}
    th = Thread(target=_run_generate_job, args=(jobId,), daemon=True)
    th.start()

    # ❗ 여기서는 절대 기다리지 않고 바로 응답
    return jsonify(ok=True, jobId=jobId, state="running")

@app.route("/status", methods=["GET"])
def status():
    jobId = request.args.get("jobId", "").strip()
    if not jobId:
        return jsonify(ok=False, error="jobId is required"), 400

    st = JOB_STATE.get(jobId)
    if not st:
        # 아직 generate()도 안 눌렀거나, 서버 재시작 등
        return jsonify(ok=True, state="none")

    resp = {
        "ok": True,
        "state": st["state"],  # "running" | "done" | "error"
    }
    if st["state"] == "done":
        resp["representative"] = st.get("rep")
    if st["state"] == "error":
        resp["error"] = st.get("error")
    return jsonify(resp)

# ③ 다운로드
@app.route("/download/<path:fname>")
def download(fname):
    # fname 예: "resultunittest001/generation/unittest001_generated.png"
    fpath = os.path.join(BASE_DIR, fname)
    if not os.path.exists(fpath):
        return jsonify(ok=False, error=f"not found: {fname}"), 404
    return send_file(fpath, mimetype="image/png")

# ④ 유사도 분석(모의)
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    filename = data.get("filename")
    jobId = data.get("jobId")

    # filename만 온 경우 fallback
    if not jobId and filename:
        # ex) "result1764687469/generation/1764687469_generated.png" 이런 식 가정
        base = os.path.basename(filename)      # 1764687469_generated.png
        jobId = base.split("_", 1)[0]          # 1764687469

    if not jobId:
        return jsonify(ok=False, error="jobId is required"), 400

    job_root = os.path.join(RESULT_DIR, f"result{jobId}")
    if not os.path.isdir(job_root):
        return jsonify(ok=False, error=f"job root not found: {job_root}"), 404

    try:
        # 첫 분석은 원본 손글씨(handwriting) 기준
        metrics = compute_similarity(
            job_id=jobId,
            out_dir=job_root,
            handwriting_subdir="handwriting",
        )
    except Exception as e:
        # 여기서 예외 잡아서 500 대신 에러 메시지 내려보내기
        return jsonify(ok=False, error=f"analyze failed: {e}"), 500

    return jsonify(ok=True, metrics=metrics)


# ⑤ 연습장(모의)
@app.route("/practice")
def practice():
    """
    연습장용 격자 이미지.
    - ?jobId=1764... 쿼리로 jobId를 받는다.
    - result{jobId}/generation 안의 {jobId}_generated_c{i}.png 를
      2x7 격자 각 칸의 '연한 배경'으로 깔아줌.
    """
    raw_jid = request.args.get("jobId", "").strip()
    if not raw_jid:
        # jobId 없으면 예전처럼 그냥 흰 캔버스 리턴
        im = Image.new("RGB", (800, 600), "white")
        buf = io.BytesIO(); im.save(buf, format="PNG"); buf.seek(0)
        return send_file(buf, mimetype="image/png")

    jobId = raw_jid
    job_root = os.path.join(RESULT_DIR, f"result{jobId}")
    gen_dir = os.path.join(job_root, "generation")

    # 기본 캔버스 크기 & 그리드 설정 (2열 x 7행)
    W, H = 800, 600
    COLS, ROWS = 2, 7
    cell_w = W // COLS
    cell_h = H // ROWS

    im = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(im)

    # 격자 라인 그리기 (연한 회색)
    line_color = (200, 200, 200)
    for r in range(ROWS + 1):
        y = r * cell_h
        draw.line([(0, y), (W, y)], fill=line_color, width=2)
    for c in range(COLS + 1):
        x = c * cell_w
        draw.line([(x, 0), (x, H)], fill=line_color, width=2)

    # 각 칸에 생성본 glyph를 연하게 깔기
    for idx, ch in enumerate(TARGET_TEXT):
        glyph_name = f"{jobId}_generated_c{idx+1}.png"
        glyph_path = os.path.join(gen_dir, glyph_name)
        if not os.path.exists(glyph_path):
            continue

        try:
            g = Image.open(glyph_path).convert("L")  # 흑백
        except Exception:
            continue

        r = idx // COLS
        c = idx % COLS
        left = c * cell_w
        top  = r * cell_h

        # 셀 크기의 70% 정도로 리사이즈
        max_w = int(cell_w * 0.7)
        max_h = int(cell_h * 0.7)
        g = g.resize((max_w, max_h), Image.LANCZOS)

        # 너무 진하지 않게: 연한 회색으로 매핑
        # (원래 0=검정, 255=흰색 -> 150~255 사이로 압축)
        g_arr = np.array(g).astype("float32")
        g_arr = 150 + (g_arr / 255.0) * 105  # 150~255
        g = Image.fromarray(g_arr.astype("uint8"))

        # 중앙 정렬 위치
        gx = left + (cell_w - max_w) // 2
        gy = top  + (cell_h - max_h) // 2

        # alpha 마스크를 이용해 살짝만 보이게 (투명도 40% 정도)
        g_rgba = g.convert("RGBA")
        alpha = int(255 * 0.4)
        r_ch, g_ch, b_ch, _ = g_rgba.split()
        new_alpha = Image.new("L", g.size, alpha)
        g_rgba = Image.merge("RGBA", (r_ch, g_ch, b_ch, new_alpha))

        im.paste(g_rgba, (gx, gy), g_rgba)

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

# app.py (추가)
@app.route("/reanalyze", methods=["POST"])
def reanalyze():
    """
    연습 페이지 전용 재검사.
    - form-data: file=<연습장 PNG>, jobId=<동일 jobId>
    - result{jobId}/retry/ 에 글자별 PNG를 저장한 뒤,
      retry 기반으로 유사도 metric 재계산.
    """
    f = request.files.get("file")
    jobId = (request.form.get("jobId") or request.args.get("jobId") or "").strip()

    if not f:
        return jsonify(ok=False, error="no file"), 400
    if not jobId:
        return jsonify(ok=False, error="jobId is required"), 400

    job_root = os.path.join(RESULT_DIR, f"result{jobId}")
    if not os.path.isdir(job_root):
        return jsonify(ok=False, error=f"job root not found: {job_root}"), 404

    retry_dir = os.path.join(job_root, "retry")
    os.makedirs(retry_dir, exist_ok=True)

    ts = str(int(time.time()))
    practice_path = os.path.join(retry_dir, f"{ts}_practice.png")
    f.save(practice_path)
    print("[REANALYZE] received practice:", practice_path)

    # 1) 연습장 이미지를 2x7 그리드로 잘라 retry 디렉토리에 유니코드 파일로 저장
    try:
        _slice_practice_to_retry(job_root, practice_path)
    except Exception as e:
        return jsonify(ok=False, error=f"slice failed: {e}"), 500

    # 2) retry 폴더 기준으로 유사도 재계산
    try:
        metrics = compute_similarity(
            job_id=jobId,
            out_dir=job_root,
            handwriting_subdir="retry",
        )
    except Exception as e:
        return jsonify(ok=False, error=f"similarity failed: {e}"), 500

    return jsonify(ok=True, metrics=metrics, practice=os.path.basename(practice_path))

def _slice_practice_to_retry(job_root: str, practice_path: str):
    """
    연습장 이미지를 2x7 그리드로 잘라서
    result{jobId}/retry 안에 각 글자를 유니코드 이름으로 저장.
    """
    retry_dir = os.path.join(job_root, "retry")
    os.makedirs(retry_dir, exist_ok=True)

    im = Image.open(practice_path).convert("L")
    W, H = im.size

    COLS, ROWS = 2, 7
    cell_w = W // COLS
    cell_h = H // ROWS

    for idx, ch in enumerate(TARGET_TEXT):
        r = idx // COLS
        c = idx % COLS

        left = c * cell_w
        top  = r * cell_h
        box  = (left, top, left + cell_w, top + cell_h)
        cell = im.crop(box)

        # 배경(연한 가이드 글자) 날리고, 사용자가 쓴 진한 획만 남기기
        arr = np.array(cell)
        # 가이드 글자는 밝게(>220) 밀어버리고, 진한 획만 남긴다
        arr = np.where(arr > 220, 255, arr)
        cell_clean = Image.fromarray(arr.astype("uint8"))

        # 기존 preprocess_char_pil 재사용해서 crop+패딩+리사이즈
        proc = preprocess_char_pil(
            cell_clean,
            img_size=IMG_SIZE,
            margin_ratio=0.10,
            binarize=True,
            thr=220,
        )

        unicode_name = f"{ord(ch)}.png"
        save_path = os.path.join(retry_dir, unicode_name)
        proc.save(save_path)
        print(f"[RETRY] saved {save_path}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
