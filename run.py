from __future__ import annotations

import os
import sys
from getpass import getpass
from pathlib import Path

import uvicorn
from openai import OpenAI
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


def save_api_key_to_env_file(api_key: str) -> None:
    line = f"OPENAI_API_KEY={api_key}\n"

    if not ENV_FILE.exists():
        ENV_FILE.write_text(line, encoding="utf-8")
        return

    lines = ENV_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
    replaced = False
    updated: list[str] = []

    for existing in lines:
        if existing.strip().startswith("OPENAI_API_KEY="):
            updated.append(line)
            replaced = True
        else:
            updated.append(existing)

    if not replaced:
        if updated and not updated[-1].endswith("\n"):
            updated[-1] = f"{updated[-1]}\n"
        updated.append(line)

    ENV_FILE.write_text("".join(updated), encoding="utf-8")


def prompt_for_api_key() -> str:
    load_dotenv(dotenv_path=ENV_FILE)

    existing = os.environ.get("OPENAI_API_KEY", "").strip()
    if existing:
        return existing

    entered = getpass("Enter your OpenAI API key: ").strip()
    if not entered:
        print("No API key provided. Exiting.")
        sys.exit(1)

    save_api_key_to_env_file(entered)
    os.environ["OPENAI_API_KEY"] = entered
    return entered


def validate_api_key(api_key: str) -> None:
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
    except Exception as exc:
        print(f"OpenAI API key validation failed: {exc}")
        sys.exit(1)


def print_quickstart() -> None:
    print("\nServer starting at http://localhost:8000")
    print("Swagger docs: http://localhost:8000/docs")
    print("\nTry these after startup:\n")
    print('curl -F "file=@samples/sample.txt" http://localhost:8000/ingest')
    print(
        "curl -X POST http://localhost:8000/ask "
        "-H \"Content-Type: application/json\" "
        "-d '{\"question\": \"What is this document about?\"}'"
    )


def main() -> None:
    api_key = prompt_for_api_key()
    validate_api_key(api_key)
    print_quickstart()

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
