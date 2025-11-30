import cv2
import numpy as np
from pathlib import Path

MIN_BLACK_RATIO = 0.005

# ⬇️ 글자 시퀀스: 반드시 grid 순서와 동일해야 함
TARGET_TEXT = "동해물과백두산이마르고닳도록"  # 예시

def get_ratio(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    _, b = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return np.sum(b == 0) / b.size

def cluster(lines, th=15):
    if not lines: return []
    lines.sort()
    cl = [[lines[0]]]
    for x in lines[1:]:
        if x - cl[-1][-1] <= th: cl[-1].append(x)
        else: cl.append([x])
    return [int(sum(c)/len(c)) for c in cl]

def parse_grid(img, save_dir, counter):
    h, w = img.shape[:2]
    edges = cv2.Canny(img, 50, 150, 3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=min(w, h)//5, maxLineGap=20)
    
    hl, vl = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            deg = abs(np.arctan2(y2-y1, x2-x1)*180/np.pi)
            if deg < 10 or deg > 170: hl.append((y1+y2)//2)
            elif 80 < deg < 100: vl.append((x1+x2)//2)

    hl, vl = cluster(hl), cluster(vl)
    if len(hl) < 2 or len(vl) < 2: return

    if 0 not in hl: hl.insert(0, 0)
    if h not in hl: hl.append(h)
    if 0 not in vl: vl.insert(0, 0)
    if w not in vl: vl.append(w)
    hl.sort(); vl.sort()

    for i in range(len(hl)-1):
        for j in range(len(vl)-1):
            y1, y2 = hl[i], hl[i+1]
            x1, x2 = vl[j], vl[j+1]
            if (y2-y1) < 20 or (x2-x1) < 20:
                continue 
            
            roi = img[y1+5:y2-5, x1+5:x2-5]

            if get_ratio(roi) >= MIN_BLACK_RATIO:

                idx = counter[0]
                if idx < len(TARGET_TEXT):
                    char = TARGET_TEXT[idx]
                    code = ord(char)  # ← 유니코드(10진)
                    filename = f"{code}.png"
                else:
                    filename = f"extra_{idx}.png"
                
                save_path = save_dir / filename
                cv2.imwrite(str(save_path), roi)
                counter[0] += 1

def run(filename, job_id=None):
    p = Path(filename)
    if not p.exists():
        print(f"[GRID] input not found: {p}")
        return

    if job_id is not None:
        out_dir = Path(f"result{job_id}")
    else:
        idx = ''.join(filter(str.isdigit, p.stem))
        out_dir = Path(f"result{idx}")

    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[GRID] failed to read image: {p}")
        return

    mid = img.shape[0] // 2
    counter = [0]

    parse_grid(img[:mid, :], out_dir, counter)
    parse_grid(img[mid:, :], out_dir, counter)

    print(f"[GRID] saved {counter[0]} cells to {out_dir}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("[GRID] usage: python grid.py <filename> [job_id]")
        sys.exit(1)

    filename = sys.argv[1]
    job_id = sys.argv[2] if len(sys.argv) >= 3 else None

    run(filename, job_id)