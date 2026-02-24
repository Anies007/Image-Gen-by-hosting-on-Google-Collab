# 🚀 Running Stable Diffusion API from Google Colab

## Overview

This system runs Stable Diffusion on Google Colab (free GPU) and exposes it as a public API using ngrok. Your local Python client sends prompts and receives generated images.

## Files Created

| File                               | Description                        |
| ---------------------------------- | ---------------------------------- |
| `colab_stable_diffusion_api.ipynb` | Colab notebook with FastAPI server |
| `image_client.py`                  | Local Python client                |
| `requirements.txt`                 | Python dependencies for client     |

---

## Step 1: Set Up Google Colab

1. **Open Google Colab**: https://colab.research.google.com

2. **Create New Notebook**: New Notebook → Python 3

3. **Enable GPU**:
   - Runtime → Change runtime type
   - Hardware accelerator: GPU
   - GPU type: T4 (or any available)

4. **Upload the notebook**:
   - File → Upload notebook
   - Select `colab_stable_diffusion_api.ipynb`

---

## Step 2: Run the Colab Server

1. **Run the cell** (press Play button or Shift+Enter)

2. **Wait for model to load** (~2-5 minutes first time)

3. **Copy the ngrok URL**:

   ```
   🎉 API Server is live at:
      https://abc1234567890.ngrok.io
   ```

4. **Keep the cell running** (don't stop it)

---

## Step 3: Install Local Client Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4: Generate Images

### Basic Usage

```bash
python image_client.py "cyberpunk city at night" --api-url https://your-ngrok-url.ngrok.io
```

### Full Example

```bash
python image_client.py \
    "a majestic lion in the savanna at sunset" \
    --api-url https://abc123.ngrok.io \
    --width 768 \
    --height 768 \
    --steps 30 \
    --seed 42
```

### Using Environment Variable

```bash
export SD_API_URL=https://abc123.ngrok.io
python image_client.py "your prompt here"
```

---

## CLI Options

| Option         | Short | Default          | Description               |
| -------------- | ----- | ---------------- | ------------------------- |
| `--api-url`    | `-u`  | (required)       | Ngrok URL from Colab      |
| `--prompt`     | -     | (required)       | Image description         |
| `--width`      | `-w`  | 512              | Image width (256-1024)    |
| `--height`     | `-h`  | 512              | Image height (256-1024)   |
| `--steps`      | `-s`  | 25               | Inference steps (1-100)   |
| `--guidance`   | `-g`  | 7.5              | Guidance scale (1-20)     |
| `--seed`       | -     | random           | Reproducibility seed      |
| `--negative`   | `-n`  | None             | Negative prompt           |
| `--timeout`    | `-t`  | 180              | Request timeout (seconds) |
| `--output-dir` | `-o`  | generated_images | Save directory            |
| `--no-save`    | -     | false            | Don't save locally        |

---

## Troubleshooting

### Connection Refused

- Make sure Colab cell is still running
- Check ngrok URL is correct
- Ngrok URL may have changed (re-run cell)

### Timeout Errors

- Increase timeout: `--timeout 300`
- Reduce steps: `--steps 15`
- Wait for Colab GPU allocation

### Out of Memory

- Reduce image size: `--width 256 --height 256`
- Reduce steps: `--steps 10`

### Ngrok Issues

- For longer sessions, add auth token in notebook
- Free ngrok URLs expire after ~2 hours

---

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  Local Client   │────────▶│   ngrok Tunnel   │────────▶│   Colab GPU     │
│ image_client.py │         │  (public URL)    │         │ Stable Diffusion│
└─────────────────┘         └──────────────────┘         └─────────────────┘
        │                                                     │
        │ POST /generate                                     │
        │ { prompt, params }                                 │
        │                                                    │
        │                    ◀──────────────                  │
        │                    { image_base64, seed }          │
        ▼                                                    ▼
┌─────────────────┐                              ┌─────────────────────┐
│ output_*.png    │                              │  PNG Image          │
│ (saved locally) │                              │  (in memory)        │
└─────────────────┘                              └─────────────────────┘
```

---

## Notes

- **Cost**: Free (Colab GPU + ngrok free tier)
- **Sessions**: Ngrok free URLs expire; re-run cell if needed
- **Privacy**: Generated images stay between your client and Colab
- **Rate Limits**: None enforced, but be reasonable
