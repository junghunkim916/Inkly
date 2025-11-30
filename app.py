from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageDraw
import os, io, time, random

from parsingmodule import parse_to_char_images
from generatemodule import generate_from_chars
from similaritymodule import compute_similarity

from threading import Thread
from generatemodule import generate_from_chars, TARGET_TEXT

JOB_STATE = {}  # { jobId: {"state": "pending|running|done|error", "rep": "...", "error": "..."} }

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)

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

        # handwriting 안의 PNG들 (0.png,1.png,... 순서라고 가정)
        all_pngs = [f for f in os.listdir(handwriting_dir) if f.lower().endswith(".png")]
        if not all_pngs:
            JOB_STATE[job_id] = {"state": "error", "error": "no handwriting images"}
            return

        # 숫자 기준 정렬
        def _nat_key(name: str):
            base = os.path.splitext(os.path.basename(name))[0]
            try:
                return int(base)
            except ValueError:
                return 10_000

        all_pngs.sort(key=_nat_key)

        # TARGET_TEXT 길이에 맞춰 자르기
        from generatemodule import TARGET_TEXT
        num_chars = min(len(all_pngs), len(TARGET_TEXT))
        char_files = all_pngs[:num_chars]

        out = generate_from_chars(
            char_files=char_files,
            out_dir=handwriting_dir,
            job_id=job_id,
            size=None
        )

        rep_name = out["representative"]  # ex) f"{job_id}_generated.png"
        rep_rel = f"result{job_id}/generation/{rep_name}"

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

    if not jobId and filename:
        jobId = os.path.basename(filename).split("_", 1)[0]
    if not jobId:
        return jsonify(ok=False, error="jobId is required"), 400

    metrics = compute_similarity(job_id=jobId, out_dir=RESULT_DIR)
    return jsonify(ok=True, metrics=metrics)

# ⑤ 연습장(모의)
@app.route("/practice")
def practice():
    # 더미 흰 캔버스
    im = Image.new("RGB", (800,600), "white")
    buf = io.BytesIO(); im.save(buf, format="PNG"); buf.seek(0)
    return send_file(buf, mimetype="image/png")

# app.py (추가)
@app.route("/reanalyze", methods=["POST"])
def reanalyze():
    """
    연습 페이지 전용: 업로드된 연습본 이미지(file)를 받아
    레이더 차트용 더미 지표를 반환한다.
    미리보기/생성 로직과 완전히 분리.
    """
    f = request.files.get("file")
    if not f:
        return jsonify(ok=False, error="no file"), 400

    ts = str(int(time.time()))
    save_path = os.path.join(RESULT_DIR, f"{ts}_practice.png")
    f.save(save_path)
    print("[REANALYZE] received practice:", save_path)

    # 더미 지표 (원하면 실제 비교 로직으로 교체)
    metrics = {
        "균형": round(random.uniform(0.6, 0.9), 2),
        "획간격": round(random.uniform(0.5, 0.9), 2),
        "기울기": round(random.uniform(0.4, 0.8), 2),
        "자간": round(random.uniform(0.5, 0.9), 2),
        "획두께": round(random.uniform(0.4, 0.9), 2)
    }
    return jsonify(ok=True, metrics=metrics, practice=os.path.basename(save_path))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
