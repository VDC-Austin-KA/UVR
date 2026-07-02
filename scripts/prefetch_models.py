"""Downloads model weights into the image at build time so the first
real request doesn't have to wait on a cold model download.

DeepFilterNet ships its checkpoint inside the pip package, so only the
UVR separation model needs fetching here.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402


def main() -> None:
    from audio_separator.separator import Separator

    print(f"Prefetching UVR model: {config.SEPARATION_MODEL}")
    separator = Separator(
        output_dir="/tmp",
        model_file_dir=str(config.MODEL_DIR),
        log_level=logging.INFO,
    )
    separator.load_model(model_filename=config.SEPARATION_MODEL)
    print("Model cached at", config.MODEL_DIR)


if __name__ == "__main__":
    main()
