from faster_whisper import WhisperModel
import torch
import datetime
import os

# --- 1. SETTINGS ---
# Put the full path to your video file here
VIDEO_PATH = "video.mp4"
# 'large-v3-turbo' is the best performance/accuracy balance for the RTX 4060
MODEL_SIZE = "large-v3-turbo"
# Set to 'en' for English, 'pt' for Portuguese, or None for auto-detect
LANGUAGE = None


def format_timestamp(seconds):
    """Converts seconds to SRT subtitle format (00:00:00,000)"""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def run_transcription():
    # Check if the video file exists at the specified path
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: The file '{VIDEO_PATH}' was not found!")
        return

    # Verify if CUDA is available and show the GPU name
    print(f"🚀 Starting with GPU: {torch.cuda.get_device_name(0)}")

    # Initialize the Whisper engine on GPU with float16 precision
    # This ensures extremely fast processing on 40-series cards
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

    print(f"🎬 Processing audio...")
    segments, info = model.transcribe(VIDEO_PATH, beam_size=5, language=LANGUAGE)

    print(f"🌍 Detected language: {info.language} ({info.language_probability:.2%})")

    # Define output file names based on the original video path
    base_name = os.path.splitext(VIDEO_PATH)[0]
    txt_file = f"{base_name}.txt"
    srt_file = f"{base_name}.srt"

    print(f"✍️ Writing results...")

    with open(txt_file, "w", encoding="utf-8") as txt, \
            open(srt_file, "w", encoding="utf-8") as srt:

        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            text_content = segment.text.strip()

            # Save to standard text file
            txt.write(f"[{start_time} -> {end_time}] {text_content}\n")

            # Save to SRT (SubRip) subtitle format
            srt.write(f"{i}\n{start_time.replace('.', ',')} --> {end_time.replace('.', ',')}\n{text_content}\n\n")

            # Print real-time progress to the console
            print(f"[{start_time}] {text_content}")

    print("\n" + "=" * 30)
    print(f"✅ COMPLETED SUCCESSFULLY!")
    print(f"📄 Text file: {txt_file}")
    print(f"🎬 Subtitle file: {srt_file}")
    print("=" * 30)


if __name__ == "__main__":
    run_transcription()