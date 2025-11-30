from PIL import Image, ImageDraw, ImageFont
import os
import time
import shutil
import subprocess
import re
from typing import List

try:
    from generatemodule import generate_from_chars, TARGET_TEXT
except Exception:
    generate_from_chars = None
    TARGET_TEXT = "동해물과백두산이마르고닳도록"  # fallback

BASE_DIR = os.path.dirname(__file__)

def _char_order_key(path: str) -> int:
    """
    result{jobId} 안의 '46028.png' 같은 유니코드 파일명을
    TARGET_TEXT 상의 순서(index)로 변환해서 정렬에 사용.
    - 파일명이 유니코드가 아니면 맨 뒤로 보냄.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        code = int(base)
        ch = chr(code)
        return TARGET_TEXT.index(ch)  # '동해물과...' 내에서 위치
    except Exception:
        return 10_000  # 정렬에서 맨 뒤로

def _run_grid_and_collect(src_path: str, grid_py_path: str, job_id: str) -> List[str]:
    """
    - src_path: 업로드 원본 (예: result{jobId}/grid_{jobId}.png)
    - grid.py 실행 후, result{jobId}/*.png 절대경로 리스트 반환
    """
    if not os.path.isfile(grid_py_path):
        raise FileNotFoundError(f"grid.py not found at: {grid_py_path}")

    grid_dir = os.path.dirname(os.path.abspath(grid_py_path))

    # grid.py 실행
    proc = subprocess.run(
        ["python", os.path.basename(grid_py_path), src_path, job_id],
        cwd=grid_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"grid.py failed (code {proc.returncode}).\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    # grid.py가 result{jobId} 안에 유니코드.png들을 저장했다고 가정
    job_root = os.path.join(BASE_DIR, f"result{job_id}")
    if not os.path.isdir(job_root):
        raise RuntimeError(f"expected folder not found: {job_root}")

    # 🔥 핵심: TARGET_TEXT 순서대로 유니코드 파일을 찾는다
    produced_pngs: List[str] = []
    for ch in TARGET_TEXT:
        fname = f"{ord(ch)}.png"
        fpath = os.path.join(job_root, fname)
        if os.path.exists(fpath):
            produced_pngs.append(fpath)
        else:
            print(f"[PARSING] warning: missing char '{ch}' ({fname}) in {job_root}")

    if not produced_pngs:
        raise RuntimeError(f"grid.py executed but no TARGET_TEXT PNG in {job_root}")

    return produced_pngs

def parse_to_char_images(
    src_path: str,
    out_dir: str | None = None,  # 🔥 이제 사실상 무시해도 됨(호환용)
    count: int = 14,
    prefix: str | None = None
):
    """
    - src_path : 업로드 원본 (보통 result{jobId}/grid_{jobId}.png)
    - prefix   : job_id (FE에서 받은 값)
    - 결과:
        result{jobId}/handwriting/0.png, 1.png, ...
        result{jobId}/generation/{jobId}_generated_c*.png, {jobId}_generated.png
    """
    ts = int(time.time())
    job_id = prefix if prefix is not None else str(ts)

    # job별 디렉토리 구성
    job_root = os.path.join(BASE_DIR, f"result{job_id}")
    handwriting_dir = os.path.join(job_root, "handwriting")
    os.makedirs(handwriting_dir, exist_ok=True)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    grid_py = os.path.join(this_dir, "grid.py")

    try:
        # 1) grid.py 실행 + result{jobId}/*.png 수집
        produced_pngs = _run_grid_and_collect(
            src_path=src_path,
            grid_py_path=grid_py,
            job_id=job_id,
        )

        # 2) count 제한
        if count is not None and count > 0:
            produced_pngs = produced_pngs[:count]

        # 3) result{jobId}/*.png → handwriting 폴더로 모으기
        char_files: List[str] = []
        for i, abs_path in enumerate(produced_pngs):
            try:
                # 파일명을 0.png, 1.png ... 로 단순화
                fname = f"{i}.png"
                dst = os.path.join(handwriting_dir, fname)
                shutil.copy2(abs_path, dst)
                char_files.append(fname)
            except Exception as e:
                print(f"[PARSING] copy failed #{i}: {e}")

        if not char_files:
            raise RuntimeError("No handwriting images saved to handwriting_dir")

    except Exception as grid_err:
        print(f"[PARSING] grid.py failed ({grid_err}). Fallback to dummy parser.")
        handwriting_dir = os.path.join(job_root, "handwriting")
        os.makedirs(handwriting_dir, exist_ok=True)

        W, H = 256, 256
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 120)
        except Exception:
            font = ImageFont.load_default()

        char_files = []
        for i in range(1, (count or 6) + 1):
            img = Image.new("RGB", (W, H), "white")
            d = ImageDraw.Draw(img)
            txt = str(i)

            bbox = d.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (W - tw) // 2
            y = (H - th) // 2
            d.text((x, y), txt, fill="black", font=font)

            fname = f"{i}.png"
            fpath = os.path.join(handwriting_dir, fname)
            img.save(fpath, format="PNG")
            char_files.append(fname)

    # ✅ 여기서 바로 보간 수행
    # if generate_from_chars is not None:
    #     try:
    #         _ = generate_from_chars(
    #             char_files=char_files,
    #             out_dir=handwriting_dir,  # 🔥 handwriting 기준
    #             job_id=job_id,
    #             size=None
    #         )
    #     except Exception as e:
    #         print(f"[PARSING] generate_from_chars failed: {e}")

    # 업로드 API는 보통 파싱된 파일 목록만 필요하므로 이것만 반환
    return char_files
