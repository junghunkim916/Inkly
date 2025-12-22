# generatemodule.py
from __future__ import annotations

import os
import math
import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
from torchvision import transforms

# ============================================================
# 0. Global config
# ============================================================

IMG_SIZE = 64
TARGET_TEXT = "동해물과백두산이마르고닳도록"  # 14 chars

try:
    BASE_DIR = os.path.dirname(__file__)
except NameError:
    BASE_DIR = os.getcwd()

CHARSET_TXT = os.path.join(BASE_DIR, "charset.txt")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_epoch_100_ema.pt")
STYLE_ENCODER_PATH = os.path.join(CHECKPOINT_DIR, "style_encoder.pt")

GOTHIC_FONT_NAME = "NanumGothic"
FONT_IMAGES_DIR = os.path.join(BASE_DIR, "font_images", GOTHIC_FONT_NAME)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_UNET = None
_STYLE_ENCODER = None
_DIFFUSION = None

# ============================================================
# 1) Charset (char -> index)
# ============================================================

def _load_chars_from_txt(path: str):
    if not os.path.exists(path):
        # fallback: TARGET_TEXT unique chars
        seen = set()
        chars = []
        for ch in TARGET_TEXT:
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)
        return chars

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = raw.replace("\n", "").replace("\r", "").replace(" ", "")
    seen = set()
    chars = []
    for ch in raw:
        if ch not in seen:
            seen.add(ch)
            chars.append(ch)
    return chars

FULL_CHARS = _load_chars_from_txt(CHARSET_TXT)
CHAR2IDX = {ch: i for i, ch in enumerate(FULL_CHARS)}

# ============================================================
# 2) Preprocess / Utils
# ============================================================

def preprocess_char_pil(img_pil, img_size=64, margin_ratio=0.10, binarize=True, thr=220):
    """
    - grayscale
    - optional binarize(white background)
    - tight crop by non-white pixels
    - pad to square
    - resize to img_size
    """
    img = img_pil.convert("L").filter(ImageFilter.MedianFilter(size=3))
    arr = np.array(img)

    if binarize:
        arr = np.where(arr > thr, 255, arr)

    rows = np.any(arr < 255, axis=1)
    cols = np.any(arr < 255, axis=0)
    if not rows.any() or not cols.any():
        return img.resize((img_size, img_size), Image.LANCZOS)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = arr.shape
    margin = int(max(rmax - rmin, cmax - cmin) * margin_ratio)
    rmin = max(0, rmin - margin); rmax = min(h, rmax + margin)
    cmin = max(0, cmin - margin); cmax = min(w, cmax + margin)

    cropped = Image.fromarray(arr).crop((cmin, rmin, cmax, rmax))

    cw, ch = cropped.size
    m = max(cw, ch)
    square = Image.new("L", (m, m), color=255)
    square.paste(cropped, ((m - cw) // 2, (m - ch) // 2))

    return square.resize((img_size, img_size), Image.LANCZOS)

def brighten_background(x, thr=0.8):
    # x in [-1,1] -> [0,1]
    x01 = (x + 1.0) / 2.0
    mask = (x01 >= thr)
    return torch.where(mask, torch.ones_like(x01), x01)

def _concat_images_row(img_paths, out_path, cell_size):
    W, H = cell_size
    canvas = Image.new("RGB", (W * len(img_paths), H), "white")
    for i, p in enumerate(img_paths):
        try:
            im = Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)
        except Exception:
            im = Image.new("RGB", (W, H), "white")
        canvas.paste(im, (i * W, 0))
    canvas.save(out_path, format="PNG")

# ============================================================
# 3) Models (UNet + Diffusion)  + StyleEncoder(✅ checkpoint-compatible)
# ============================================================

class StyleEncoder(nn.Module):
    """
    ✅ MUST match checkpoint keys: conv1..conv4, fc.0, fc.0.bias ...
    """
    def __init__(self, style_dim=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        # checkpoint has fc.0.*
        self.fc = nn.Sequential(
            nn.Linear(512, style_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, style_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.style_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(style_dim, out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb, s_emb):
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = h + self.style_mlp(s_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)

class UNet(nn.Module):
    def __init__(self, img_channels=1, base_channels=64, channel_mults=(1,2,4,8),
                 num_res_blocks=2, time_emb_dim=256, style_dim=512, num_chars=14):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.char_emb = nn.Embedding(num_chars, style_dim)
        self.null_token = nn.Parameter(torch.randn(style_dim))

        channels = [base_channels * m for m in channel_mults]
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for ch in channels:
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_ch, ch, time_emb_dim, style_dim))
                in_ch = ch
            self.down_blocks.append(blocks)

        self.down_samples = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, stride=2, padding=1) for ch in channels[:-1]
        ])

        self.mid_block1 = ResidualBlock(channels[-1], channels[-1], time_emb_dim, style_dim)
        self.mid_block2 = ResidualBlock(channels[-1], channels[-1], time_emb_dim, style_dim)

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        channels_reversed = list(reversed(channels))
        for i, ch in enumerate(channels_reversed):
            blocks = nn.ModuleList()
            out_ch = channels_reversed[i + 1] if i < len(channels_reversed) - 1 else base_channels
            for j in range(num_res_blocks + 1):
                in_ch_block = ch * 2 if j == 0 else ch
                blocks.append(ResidualBlock(in_ch_block, ch, time_emb_dim, style_dim))
            self.up_blocks.append(blocks)
            if i < len(channels_reversed) - 1:
                self.up_samples.append(nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1))

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, 3, padding=1),
        )

    def forward(self, x, t, char_idx, style_emb=None, use_null_token=False):
        t_emb = self.time_mlp(t)

        if use_null_token:
            s_emb = self.null_token.unsqueeze(0).expand(x.size(0), -1)
        elif style_emb is not None:
            s_emb = style_emb + self.char_emb(char_idx)
        else:
            s_emb = self.char_emb(char_idx)

        x = self.init_conv(x)

        skips = []
        for si, blocks in enumerate(self.down_blocks[:-1]):
            for block in blocks:
                x = block(x, t_emb, s_emb)
            skips.append(x)
            x = self.down_samples[si](x)

        for block in self.down_blocks[-1]:
            x = block(x, t_emb, s_emb)
        skips.append(x)

        x = self.mid_block1(x, t_emb, s_emb)
        x = self.mid_block2(x, t_emb, s_emb)

        for ui, blocks in enumerate(self.up_blocks[:-1]):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = blocks[0](x, t_emb, s_emb)
            for block in blocks[1:]:
                x = block(x, t_emb, s_emb)
            x = self.up_samples[ui](x)

        skip = skips.pop()
        x = torch.cat([x, skip], dim=1)
        x = self.up_blocks[-1][0](x, t_emb, s_emb)
        for block in self.up_blocks[-1][1:]:
            x = block(x, t_emb, s_emb)

        return self.final_conv(x)

class DiffusionModel:
    def __init__(self, timesteps=1000, device=DEVICE):
        self.timesteps = timesteps
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        steps = timesteps + 1
        s = 0.008
        x = torch.linspace(0, timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x_0 + sqrt_om * noise

# ============================================================
# 4) Batch parallel sampling (noise blending)
# ============================================================

def noise_blending_interpolation_batch(
    model: UNet,
    style_encoder: StyleEncoder,
    handwriting_img: torch.Tensor,   # (B,1,H,W)
    gothic_img: torch.Tensor,        # (B,1,H,W)
    char_idx: torch.Tensor,          # (B,)
    diffusion: DiffusionModel,
    device: torch.device,
    lambda_val=0.6,
    guid_hand_early=6.5,
    guid_goth_late=6.0,
    guid_hand_late=2.5,
    guid_goth_early=3.0,
    ramp_power=2.5,
    t_start_frac=0.6,
    variance_scale=0.0,
    sampling_stride=1,
):
    model.eval()
    with torch.no_grad():
        B = handwriting_img.size(0)

        style_hand = style_encoder(handwriting_img)  # (B,512)
        style_goth = style_encoder(gothic_img)       # (B,512)

        T = diffusion.timesteps
        t_start = int(max(1, min(T - 1, round(t_start_frac * (T - 1)))))

        noise = torch.randn_like(handwriting_img)
        t0 = torch.full((B,), t_start, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(handwriting_img, t0, noise=noise)

        for t in range(t_start, -1, -sampling_stride):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            progress = t / float(T - 1)
            sched_goth = (1.0 - progress) ** ramp_power
            w_goth = (1.0 - lambda_val) * sched_goth
            w_hand = 1.0 - w_goth

            g_hand = guid_hand_early * progress + guid_hand_late * (1.0 - progress)
            g_goth = guid_goth_early * progress + guid_goth_late * (1.0 - progress)

            hand_c = model(x_t, t_tensor, char_idx, style_hand, use_null_token=False)
            hand_u = model(x_t, t_tensor, char_idx, None,       use_null_token=True)
            goth_c = model(x_t, t_tensor, char_idx, style_goth, use_null_token=False)
            goth_u = model(x_t, t_tensor, char_idx, None,       use_null_token=True)

            noise_hand = hand_u + g_hand * (hand_c - hand_u)
            noise_goth = goth_u + g_goth * (goth_c - goth_u)
            noise_pred = w_hand * noise_hand + w_goth * noise_goth

            beta_t  = diffusion.betas[t].view(1, 1, 1, 1)
            sqrt_1m = diffusion.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1, 1)
            sqrt_ra = torch.sqrt(1.0 / diffusion.alphas[t]).view(1, 1, 1, 1)

            model_mean = sqrt_ra * (x_t - beta_t * noise_pred / sqrt_1m)
            model_mean = torch.clamp(model_mean, -1.0, 1.0)

            if t > 0:
                var = diffusion.posterior_variance[t].view(1, 1, 1, 1)
                x_t = model_mean + torch.sqrt(var * variance_scale) * torch.randn_like(x_t)
            else:
                x_t = model_mean

        return x_t

# ============================================================
# 5) Load checkpoints (robust)
# ============================================================

def _strip_module_prefix(state: dict) -> dict:
    return {k.replace("module.", ""): v for k, v in state.items()}

def _pick_state_dict(obj):
    """
    handle:
      - raw state_dict
      - {"state_dict": ...}
      - {"model_state_dict": ...}
      - {"model_ema": ...}
    """
    if not isinstance(obj, dict):
        return obj
    if "model_ema" in obj and isinstance(obj["model_ema"], dict):
        return obj["model_ema"]
    if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        return obj["model_state_dict"]
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj

def _load_models():
    global _UNET, _STYLE_ENCODER, _DIFFUSION
    if _UNET is not None and _STYLE_ENCODER is not None and _DIFFUSION is not None:
        return

    # ---- Style encoder ----
    se = StyleEncoder(style_dim=512).to(DEVICE)
    if os.path.exists(STYLE_ENCODER_PATH):
        ckpt = torch.load(STYLE_ENCODER_PATH, map_location=DEVICE)
        state = _pick_state_dict(ckpt)
        if isinstance(state, dict):
            state = _strip_module_prefix(state)
        se.load_state_dict(state, strict=True)
    se.eval()

    # ---- UNet ----
    unet = UNet(
        img_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        style_dim=512,
        num_chars=len(FULL_CHARS),
    ).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        state = _pick_state_dict(ckpt)
        if isinstance(state, dict):
            state = _strip_module_prefix(state)
        unet.load_state_dict(state, strict=True)
    unet.eval()

    diffusion = DiffusionModel(timesteps=1000, device=DEVICE)

    _STYLE_ENCODER = se
    _UNET = unet
    _DIFFUSION = diffusion

# ============================================================
# 6) Main API
# ============================================================

def generate_from_chars(char_files, out_dir, job_id: str, size=(256, 256)):
    """
    INPUT (expected):
      result{jobId}/handwriting/0.png ~ 13.png

    OUTPUT:
      result{jobId}/generation/0.png ~ 13.png (always)
      + 2 representative images:
        - {jobId}_handwriting_concat.png
        - {jobId}_generated_concat.png

    RETURN:
      representative (generated concat)
      representative_original (handwriting concat)
      handwriting (list of relative paths)
      generated (list of relative paths)
    """
    _load_models()

    handwriting_dir = out_dir
    result_root = os.path.dirname(handwriting_dir.rstrip(os.sep))
    gen_dir = os.path.join(result_root, "generation")
    os.makedirs(gen_dir, exist_ok=True)

    W, H = size if size is not None else (256, 256)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [0,1] -> [-1,1]
    ])

    # --- load batches (always 14, missing -> blank white) ---
    hw_tensors = []
    goth_tensors = []
    char_indices = []
    hw_paths = []
    gen_paths = []

    for i, ch in enumerate(TARGET_TEXT):
        # handwriting path (0.png..13.png)
        hw_path = os.path.join(handwriting_dir, f"{i}.png")
        if os.path.exists(hw_path):
            hw_raw = Image.open(hw_path).convert("L")
            hw_proc = preprocess_char_pil(hw_raw, IMG_SIZE, 0.10, True, 220)
        else:
            # blank fallback
            hw_proc = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)

        hw_tensors.append(tf(hw_proc))
        hw_paths.append(hw_path if os.path.exists(hw_path) else None)

        # gothic reference (unicode png)
        goth_path = os.path.join(FONT_IMAGES_DIR, f"{ord(ch)}.png")
        if os.path.exists(goth_path):
            goth_raw = Image.open(goth_path).convert("L")
            goth_proc = preprocess_char_pil(goth_raw, IMG_SIZE, 0.10, False, 220)
        else:
            goth_proc = hw_proc

        goth_tensors.append(tf(goth_proc))
        char_indices.append(CHAR2IDX.get(ch, 0))

    hw_batch = torch.stack(hw_tensors, dim=0).to(DEVICE)         # (14,1,64,64)
    goth_batch = torch.stack(goth_tensors, dim=0).to(DEVICE)     # (14,1,64,64)
    char_idx = torch.tensor(char_indices, device=DEVICE, dtype=torch.long)  # (14,)

    with torch.no_grad():
        result_batch = noise_blending_interpolation_batch(
            _UNET,
            _STYLE_ENCODER,
            hw_batch,
            goth_batch,
            char_idx,
            _DIFFUSION,
            DEVICE,
            lambda_val=0.6,
            sampling_stride=1,
            variance_scale=0.0,
        )
        x01 = brighten_background(result_batch, thr=0.8)  # (14,1,64,64) in [0,1]

    # --- save generation/0..13.png always ---
    for i in range(len(TARGET_TEXT)):
        arr = (x01[i, 0].detach().cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").convert("RGB").resize((W, H), Image.LANCZOS)
        out_path = os.path.join(gen_dir, f"{i}.png")
        img.save(out_path, format="PNG")
        gen_paths.append(out_path)

    # --- 대표 이미지 2개 ---
    rep_hand = os.path.join(gen_dir, f"{job_id}_handwriting_concat.png")
    rep_gen  = os.path.join(gen_dir, f"{job_id}_generated_concat.png")

    # handwriting 대표는 실제 파일 없던 칸은 흰이미지로 채움
    hw_for_concat = []
    for i in range(len(TARGET_TEXT)):
        p = os.path.join(handwriting_dir, f"{i}.png")
        if os.path.exists(p):
            hw_for_concat.append(p)
        else:
            tmp = os.path.join(gen_dir, f"__blank_hw_{i}.png")
            Image.new("RGB", (W, H), "white").save(tmp, format="PNG")
            hw_for_concat.append(tmp)

    _concat_images_row(hw_for_concat, rep_hand, (W, H))
    _concat_images_row(gen_paths, rep_gen, (W, H))

    # --- return relative paths for /download/<path:fname> ---
    rep_gen_rel  = f"{os.path.basename(rep_gen)}"
    rep_hand_rel = f"{os.path.basename(rep_hand)}"

    handwriting_rel = [f"result{job_id}/handwriting/{i}.png" for i in range(len(TARGET_TEXT))]
    generated_rel   = [f"result{job_id}/generation/{i}.png" for i in range(len(TARGET_TEXT))]

    return {
        "representative": rep_gen_rel,
        "representative_original": rep_hand_rel,
        "handwriting": handwriting_rel,
        "generated": generated_rel,
    }
