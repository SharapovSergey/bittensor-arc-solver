"""
Prep phase — выполняется С интернетом.
Загружаем Qwen2.5-72B из HuggingFace на диск валидатора.
"""

import os
import sys

model_name = "Qwen/Qwen2.5-72B-Instruct"


def main():
    print(f"📥 Downloading model: {model_name}")

    try:
        from huggingface_hub import snapshot_download
        save_dir = os.environ.get("MODEL_SAVE_DIR", "/app/models")
        os.makedirs(save_dir, exist_ok=True)

        path = snapshot_download(
            repo_id=model_name,
            local_dir=f"{save_dir}/{model_name.replace('/', '_')}",
            ignore_patterns=["*.gguf", "*.bin"],  # keep only safetensors
        )
        print(f"✅ Model downloaded to: {path}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
