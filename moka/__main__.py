import argparse
import json
import os
from pathlib import Path
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Moka local animation workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    config = Path(__file__).resolve().parents[1]/".moka/engines.json"
    if config.is_file():
        for key, value in json.loads(config.read_text("utf-8")).items():
            if key in ("MOKA_SEETHROUGH_HOME", "MOKA_SEETHROUGH_PYTHON", "MOKA_AI_PYTHON", "MOKA_AI_ENGINES"):
                os.environ.setdefault(key, str(value))
    uvicorn.run("moka.server:app", host=args.host, port=args.port, log_level="info")

if __name__ == "__main__": main()
