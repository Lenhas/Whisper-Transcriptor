# 🚀 AI Video Transcriptor (Optimized for RTX 4060)

This project uses **Faster-Whisper** to transcribe videos locally with high precision and speed, leveraging the power of NVIDIA's RTX 40 series GPUs.

## 🛠️ Tech Stack
* **Python 3.12**
* **Faster-Whisper** (Large-v3-Turbo model)
* **CUDA 12.4** (GPU Acceleration)
* **PyTorch**

## ⚡ Performance
Thanks to the **RTX 4060 (8GB VRAM)**, this script can transcribe a 10-minute video in less than 40 seconds using the `large-v3-turbo` model with `float16` precision.

## 📁 Features
- Automatic language detection.
- Generates standard text files (`.txt`).
- Generates subtitle files (`.srt`) ready for video players.
- Fully offline and private.

## 🚀 How to use
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Set your video path in the `.env` file.
4. Run `python main.py`.
