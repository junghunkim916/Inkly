from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ============================================================
# 0. 전역 설정
# ============================================================

# TARGET_TEXT = "동해물과백두산이마르고닳도록"   # 14글자
IMG_SIZE = 64                                   # 모델 인풋 해상도

BASE_DIR = os.path.dirname(__file__)
CHARSET_TXT = os.path.join(BASE_DIR, "charset.txt")

# 체크포인트 / 폰트 이미지 경로 (폴더 구조에 따라 수정 가능)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_epoch_70_ema.pt")   # ★ 네가 말한 모델
STYLE_ENCODER_PATH = os.path.join(CHECKPOINT_DIR, "style_encoder.pt") # style encoder

GOTHIC_FONT_NAME = "NanumGothic"
FONT_IMAGES_DIR = os.path.join(BASE_DIR, "font_images", GOTHIC_FONT_NAME)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# lazy loading용 전역 변수
_UNET = None
_STYLE_ENCODER = None
_DIFFUSION = None


def load_chars_from_txt(path):
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

FULL_CHARS = load_chars_from_txt(CHARSET_TXT)
CHAR2IDX = {ch: i for i, ch in enumerate(FULL_CHARS)}

TARGET_TEXT = "동해물과백두산이마르고닳도록"  # 여긴 그대로, '동해물..'만 써먹을거야

# ============================================================
# 1. 전처리 함수 (preprocess_char_pil)
# ============================================================

def preprocess_char_pil(img_pil, img_size=64, margin_ratio=0.10, binarize=True, thr=220):
    """
    손글씨/스캔 이미지에도 고딕과 동일하게 크롭-패딩-리사이즈 적용.
    - 배경이 완전 흰색이 아닐 수 있으니 간단 이진화(thr) 옵션 제공
    """
    img = img_pil.convert('L').filter(ImageFilter.MedianFilter(size=3))
    arr = np.array(img)

    if binarize:
        # 배경 밝게, 획 어둡게 가정: 임계값으로 배경을 255로 밀어줌
        arr = np.where(arr > thr, 255, arr)

    # 글자 영역 찾기 (흰색이 아닌 곳)
    rows = np.any(arr < 255, axis=1)
    cols = np.any(arr < 255, axis=0)
    if not rows.any() or not cols.any():
        # 비어있으면 그대로 리사이즈만
        return img.resize((img_size, img_size), Image.LANCZOS)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # 10% 여백
    h, w = arr.shape
    margin = int(max(rmax - rmin, cmax - cmin) * margin_ratio)
    rmin = max(0, rmin - margin); rmax = min(h, rmax + margin)
    cmin = max(0, cmin - margin); cmax = min(w, cmax + margin)

    cropped = Image.fromarray(arr).crop((cmin, rmin, cmax, rmax))

    # 정사각 패딩
    cw, ch = cropped.size
    m = max(cw, ch)
    square = Image.new('L', (m, m), color=255)
    square.paste(cropped, ((m - cw)//2, (m - ch)//2))

    # 최종 리사이즈
    return square.resize((img_size, img_size), Image.LANCZOS)


def brighten_background(x, thr=0.8):
    """
    x : (B,1,H,W), [-1,1] 범위 텐서 (DDPM 결과)
    thr : [0,1] 기준 임계값
          - x01 >= thr 인 픽셀은 전부 1.0(흰색)으로 올림
          - 나머지는 원래 값 유지
    """
    # [-1,1] -> [0,1]
    x01 = (x + 1.0) / 2.0  # (B,1,H,W)

    # 밝은 픽셀 마스크 (배경 후보)
    mask = (x01 >= thr).float()

    # 밝은 곳은 1.0, 나머지는 그대로
    x01_clean = x01 * (1.0 - mask) + mask * 1.0

    return x01_clean   # [0,1] 범위 반환


def brighten_and_upscale(x, thr=0.7, scale=4):
    """
    x : (B,1,H,W), [-1,1]
    반환: (B, H*scale, W*scale)  numpy 배열 (0~1)
    """
    x01 = brighten_background(x, thr=thr)  # [0,1]

    imgs = []
    for i in range(x01.size(0)):
        arr = x01[i, 0].detach().cpu().numpy()   # (H,W), 0~1
        pil = Image.fromarray((arr * 255).astype(np.uint8))
        up = pil.resize(
            (pil.width * scale, pil.height * scale),
            Image.BILINEAR
        )
        up_arr = np.asarray(up).astype(np.float32) / 255.0
        imgs.append(up_arr)

    # (B, H', W') numpy 리스트
    return imgs


# ============================================================
# 2. 모델 정의 (StyleEncoder, UNet, DiffusionModel)
# ============================================================

class StyleEncoder(nn.Module):
    # input: 1x64x64 / output: 512-dim style vector
    def __init__(self, style_dim=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 64 -> 32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 32 -> 16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 16 -> 8
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
        x = self.fc(x)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, style_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.style_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(style_dim, out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb, s_emb):
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = h + self.style_mlp(s_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, img_channels=1, base_channels=64, channel_mults=(1, 2, 4, 8),
                 num_res_blocks=2, time_emb_dim=256, style_dim=512, num_chars=14):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
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
            nn.Conv2d(base_channels, img_channels, 3, padding=1)
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
    """DDPM with Cosine Schedule (모든 텐서를 같은 device에서 생성)"""
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        # 여기: device가 torch.device든 문자열이든 그냥 통일
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        steps = timesteps + 1
        s = 0.008

        # 처음부터 device 위에서 생성
        x = torch.linspace(0, timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)

        # 🔴 여기 세 줄이 중요
        self.betas = betas.to(self.device)                    # (T,)
        self.alphas = (1.0 - betas).to(self.device)           # (T,)
        self.alphas_cumprod = torch.cumprod(
            self.alphas, dim=0
        ).to(self.device)                                     # (T,)

        self.alphas_cumprod_prev = torch.cat(
            [
                torch.ones(1, device=self.device),
                self.alphas_cumprod[:-1]
            ],
            dim=0,
        )   # 둘 다 같은 device

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # 필요하면 여기도 .to(self.device) 붙여도 됨
        # self.posterior_variance = self.posterior_variance.to(self.device)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
# ============================================================
# 3. Noise-blending interpolation (inference)
# ============================================================

def noise_blending_interpolation(
    model,
    style_encoder,
    handwriting_img,
    gothic_img,
    char_idx,
    diffusion,
    device,
    lambda_val=0.6,
    guid_hand_early=6.5,
    guid_goth_late=6.0,
    guid_hand_late=2.5,
    guid_goth_early=3.0,
    ramp_power=2.5,
    t_start_frac=0.6,
    variance_scale=0.8,
    sampling_stride=1,
):
    """
    - 초반(t가 큼): 손글씨 비중 / 후반(t가 작): 고딕 비중
    - t_start에서 image-to-image 시작
    """
    model.eval()
    with torch.no_grad():
        style_hand = style_encoder(handwriting_img)
        style_goth = style_encoder(gothic_img)
        char_idx_tensor = torch.tensor([char_idx], device=device)

        T = diffusion.timesteps
        t_start = int(max(1, min(T - 1, round(t_start_frac * (T - 1)))))

        noise = torch.randn_like(handwriting_img)
        t0 = torch.full((handwriting_img.size(0),), t_start, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(handwriting_img, t0, noise=noise)

        for t in range(t_start, -1, -sampling_stride):
            t_tensor = torch.tensor([t], device=device)

            progress = t / float(T - 1)
            sched_goth = (1.0 - progress) ** ramp_power
            w_goth = (1.0 - lambda_val) * sched_goth
            w_hand = 1.0 - w_goth

            g_hand = guid_hand_early * progress + guid_hand_late * (1.0 - progress)
            g_goth = guid_goth_early * progress + guid_goth_late * (1.0 - progress)

            hand_c = model(x_t, t_tensor, char_idx_tensor, style_hand, use_null_token=False)
            hand_u = model(x_t, t_tensor, char_idx_tensor, None,       use_null_token=True)
            goth_c = model(x_t, t_tensor, char_idx_tensor, style_goth, use_null_token=False)
            goth_u = model(x_t, t_tensor, char_idx_tensor, None,       use_null_token=True)

            noise_hand = hand_u + g_hand * (hand_c - hand_u)
            noise_goth = goth_u + g_goth * (goth_c - goth_u)
            noise_pred = w_hand * noise_hand + w_goth * noise_goth

            beta_t  = diffusion.betas[t].view(-1, 1, 1, 1)
            sqrt_1m = diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_ra = torch.sqrt(1.0 / diffusion.alphas[t]).view(-1, 1, 1, 1)

            model_mean = sqrt_ra * (x_t - beta_t * noise_pred / sqrt_1m)
            model_mean = torch.clamp(model_mean, -1.0, 1.0)

            if t > 0:
                var = diffusion.posterior_variance[t].view(-1, 1, 1, 1)
                x_t = model_mean + torch.sqrt(var * variance_scale) * torch.randn_like(x_t)
            else:
                x_t = model_mean

        return x_t


# ============================================================
# 4. 모델 로더 (lazy load)
# ============================================================

def _load_models():
    global _UNET, _STYLE_ENCODER, _DIFFUSION

    if _UNET is not None and _STYLE_ENCODER is not None and _DIFFUSION is not None:
        return

    # Style Encoder
    style_encoder = StyleEncoder(style_dim=512).to(DEVICE)
    if os.path.exists(STYLE_ENCODER_PATH):
        ckpt_se = torch.load(STYLE_ENCODER_PATH, map_location=DEVICE)
        style_encoder.load_state_dict(ckpt_se)
    style_encoder.eval()

    # U-Net
    unet = UNet(
        img_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        style_dim=512,
        # num_chars=len(TARGET_TEXT)
        num_chars=len(FULL_CHARS)
    ).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(ckpt, dict):
            if "model_ema" in ckpt:
                state = ckpt["model_ema"]
            elif "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            else:
                state = ckpt
        else:
            state = ckpt
        unet.load_state_dict(state, strict=True)
    unet.eval()

    diffusion = DiffusionModel(timesteps=1000, device=DEVICE)

    _UNET = unet
    _STYLE_ENCODER = style_encoder
    _DIFFUSION = diffusion


# ============================================================
# 5. 메인 API 함수: generate_from_chars
# ============================================================
def generate_from_chars(char_files, out_dir, job_id: str, size=(256, 256)):
    """
    ...
    """
    handwriting_dir = out_dir

    result_root = os.path.dirname(handwriting_dir.rstrip(os.sep))
    gen_dir = os.path.join(result_root, "generation")
    os.makedirs(gen_dir, exist_ok=True)

    _load_models()

    # ---------- size 처리 ----------
    if size is None:
        W = H = None
        for fname in char_files:
            src_path = os.path.join(handwriting_dir, fname)
            if os.path.exists(src_path):
                with Image.open(src_path) as im:
                    W, H = im.size
                break
        if W is None or H is None:
            W, H = 256, 256
    else:
        W, H = size
    # ------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [0,1] -> [-1,1]
    ])
    partials = []
    num_chars = min(len(char_files), len(TARGET_TEXT))

    for i in range(num_chars):
        char = TARGET_TEXT[i]               # '동', '해', ...
        src_fname = char_files[i]
        src_path = os.path.join(handwriting_dir, src_fname)

        if not os.path.exists(src_path):
            continue

        # 1) 손글씨 전처리
        hw_raw = Image.open(src_path).convert("L")
        hw_proc = preprocess_char_pil(
            hw_raw,
            img_size=IMG_SIZE,
            margin_ratio=0.10,
            binarize=True,
            thr=220
        )
        hw_tensor = transform(hw_proc).unsqueeze(0).to(DEVICE)

        # 2) 고딕 reference
        goth_path = os.path.join(FONT_IMAGES_DIR, f"{ord(char)}.png")
        if os.path.exists(goth_path):
            goth_raw = Image.open(goth_path).convert("L")
            goth_proc = preprocess_char_pil(
                goth_raw,
                img_size=IMG_SIZE,
                margin_ratio=0.10,
                binarize=False
            )
            goth_tensor = transform(goth_proc).unsqueeze(0).to(DEVICE)
        else:
            goth_tensor = hw_tensor.clone()

        # 3) Noise Blending
        try:
            # 🔴 fsid.py 와 동일하게, charset.txt 기준 인덱스 사용
            char_idx = CHAR2IDX[char]
        except KeyError:
            # charset.txt 에 없는 글자면 걍 0번으로 fallback (또는 continue 해도 됨)
            char_idx = 0

        with torch.no_grad():
            result_tensor = noise_blending_interpolation(
                _UNET,
                _STYLE_ENCODER,
                hw_tensor,
                goth_tensor,
                char_idx,
                _DIFFUSION,
                DEVICE,
                lambda_val=0.6,
                sampling_stride=1,
                variance_scale=0.0,
            )

            x01 = brighten_background(result_tensor, thr=0.8)   # (B,1,H,W), [0,1]
            arr = (x01[0, 0].cpu().numpy() * 255).astype(np.uint8)
            res_img = Image.fromarray(arr, mode="L").convert("RGB")

        if res_img.size != (W, H):
            res_img = res_img.resize((W, H), Image.LANCZOS)

        out_name = f"{job_id}_generated_c{i+1}.png"
        out_path = os.path.join(gen_dir, out_name)
        res_img.save(out_path, format="PNG")
        partials.append(out_name)

    # # 5) representative 이어붙이기
    # if len(partials) > 0:
    #     rep_width = W * len(partials)
    #     rep_height = H
    #     rep = Image.new("RGB", (rep_width, rep_height), "white")

    #     for idx, fname in enumerate(partials):
    #         p_path = os.path.join(gen_dir, fname)
    #         glyph_img = Image.open(p_path).convert("RGB")
    #         glyph_img = glyph_img.resize((W, H), Image.LANCZOS)
    #         rep.paste(glyph_img, (idx * W, 0))

    #     rep_name = f"{job_id}_generated_c1.png"
    #     rep.save(os.path.join(gen_dir, rep_name), format="PNG")
    # else:
    #     rep = Image.new("RGB", (W, H), "white")
    #     draw = ImageDraw.Draw(rep)
    #     try:
    #         font = ImageFont.truetype("DejaVuSans.ttf", 72)
    #     except Exception:
    #         font = ImageFont.load_default()
    #     text = "result"
    #     bbox = draw.textbbox((0, 0), text, font=font)
    #     tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    #     draw.text(((W - tw) // 2, (H - th) // 2), text, fill="black", font=font)
    #     rep_name = f"{job_id}_generated_c1.png"
    #     rep.save(os.path.join(gen_dir, rep_name), format="PNG")

    # return {"representative": rep_name, "partials": partials}
    #시연용
        # 5) 대표 이미지: 원본 동 + 개선된 동 나란히 비교
    if len(partials) > 0:
        original_path = os.path.join(handwriting_dir, char_files[0])
        try:
            # 1) 원본 불러와서
            orig_raw = Image.open(original_path).convert("L")
            # 2) 모델에 넣는 것과 동일한 전처리 적용
            orig_proc = preprocess_char_pil(
                orig_raw,
                img_size=IMG_SIZE,
                margin_ratio=0.10,
                binarize=True,
                thr=220,
            )
            # 3) 최종 프리뷰 크기로 리사이즈 후 RGB 변환
            if orig_proc.size != (W, H):
                orig_proc = orig_proc.resize((W, H), Image.LANCZOS)
            orig_img = orig_proc.convert("RGB")
        except Exception:
            orig_img = Image.new("RGB", (W, H), "white")

        # 2) 개선된 동 (첫 번째 생성 결과 partials[0])
        gen_path = os.path.join(gen_dir, partials[0])
        try:
            gen_img = Image.open(gen_path).convert("RGB")
        except Exception:
            gen_img = Image.new("RGB", (W, H), "white")

        if gen_img.size != (W, H):
            gen_img = gen_img.resize((W, H), Image.LANCZOS)

        # 3) 두 이미지를 가로로 이어붙인 비교 이미지
        rep_width = W * 2
        rep_height = H
        rep = Image.new("RGB", (rep_width, rep_height), "white")
        # 왼쪽: 원본 손글씨 동
        rep.paste(orig_img, (0, 0))
        # 오른쪽: 개선된 동
        rep.paste(gen_img, (W, 0))

        rep_name = f"{job_id}_compare.png"
        rep.save(os.path.join(gen_dir, rep_name), format="PNG")
    else:
        # 생성 결과가 하나도 없는 예외 상황
        rep_width = W * 2
        rep_height = H
        rep = Image.new("RGB", (rep_width, rep_height), "white")
        draw = ImageDraw.Draw(rep)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 72)
        except Exception:
            font = ImageFont.load_default()
        text = "result"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((rep_width - tw) // 2, (rep_height - th) // 2),
                  text, fill="black", font=font)
        rep_name = f"{job_id}_compare.png"
        rep.save(os.path.join(gen_dir, rep_name), format="PNG")
    return {"representative": rep_name, "partials": partials}
