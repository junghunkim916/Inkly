import os, time
from datetime import datetime
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
from PIL import Image
import numpy as np

BASE_DIR   = "/content"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

API_KEY = os.environ.get("INKLY_API_KEY", "dev-key")
def check_key(req): return req.headers.get("X-API-Key") == API_KEY

app = Flask(__name__)
CORS(app)

@app.post("/upload")
def upload():
    if not check_key(request): return ("Forbidden", 403)
    if "file" not in request.files: return jsonify({"ok": False, "error": "file field required"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".png"):
        return jsonify({"ok": False, "error": "PNG only"}), 400
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{f.filename}"
    f.save(os.path.join(UPLOAD_DIR, fname))
    return jsonify({"ok": True, "filename": fname})

def fake_infer(pil_img, lam: float):
    arr = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    arr = np.clip(arr ** (1.2 - 0.2*lam), 0, 1)
    return Image.fromarray((arr*255).astype(np.uint8))

@app.post("/generate")
def generate():
    if not check_key(request): return ("Forbidden", 403)
    data = request.get_json(force=True, silent=True) or {}
    filename = data.get("filename")
    lambdas  = data.get("lambdas", [0.2,0.4,0.6,0.8,1.0])
    job_id   = str(data.get("jobId", int(time.time())))
    if not filename: return jsonify({"ok": False, "error": "filename required"}), 400

    src = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(src): return jsonify({"ok": False, "error": "file not found"}), 404
    img = Image.open(src)

    out_dir = os.path.join(RESULT_DIR, job_id); os.makedirs(out_dir, exist_ok=True)
    outputs = []
    for lam in lambdas:
        out_name = f"sentence_lambda_{lam:.1f}.png"
        out_path = os.path.join(out_dir, out_name)
        fake_infer(img, lam).save(out_path)
        outputs.append(f"{job_id}/{out_name}")

    rep = f"{job_id}/sentence_lambda_0.6.png"
    return jsonify({"ok": True, "jobId": job_id, "results": outputs, "representative": rep})

@app.get("/download/<path:subpath>")
def download(subpath):
    if not check_key(request): return ("Forbidden", 403)
    full = os.path.join(RESULT_DIR, subpath)
    if not os.path.exists(full): abort(404)
    return send_file(full, mimetype="image/png", as_attachment=False)

if __name__ == "__main__":
    from waitress import serve
    print("Running on :8000")
    serve(app, host="0.0.0.0", port=8000)