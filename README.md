# Inkly
# Inkly Model Server (Colab)

AI handwriting style transfer server running on Google Colab + Cloudflare Tunnel.
**checkpoints 폴더에 model을 upload해야 합니다.

## Features
- Grid-based handwriting segmentation
- Style encoder + diffusion-based generator
- Similarity analysis
- REST API (Flask)

## API
### POST /upload
Upload grid image → parse → return character list

### POST /generate
Trigger handwriting generation

### GET /status?jobId=xxxx
Check generation status

### GET /download/{path}
Download generated result

## Running (Colab)
1. Upload this repository to Colab
2. Run:
