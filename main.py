#!/usr/bin/env python3
"""
Batch Image Captioning Tool
Sends images to an OpenAI-compatible vision endpoint and saves captions
to a .txt file in each processed directory (or alongside listed files).
"""

import os
import sys
import json
import base64
import argparse
import requests
from pathlib import Path
from typing import List, Optional


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }
    return mime_map.get(suffix, "application/octet-stream")


def send_caption_request(
    api_url: str,
    api_key: str,
    image_path: Path,
    model: str,
    prompt: Optional[str] = None,
    timeout: int = 120,
) -> dict:
    """
    Send a single image to the OpenAI-compatible vision endpoint.
    Returns the parsed JSON response.
    """
    b64_data = encode_image_to_base64(str(image_path))
    mime_type = get_mime_type(image_path)

    if prompt is None:
        prompt = "Describe this image in detail."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_data}"
                    },
                },
            ],
        }
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def extract_caption(response_json: dict) -> str:
    """Extract the caption text from an OpenAI-compatible response."""
    try:
        choices = response_json.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        # Handle both string and list-of-blocks content formats
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
            return "\n".join(parts).strip()
    except Exception as e:
        print(f"  [WARN] Failed to parse response: {e}")
    return ""


def collect_images_from_directory(directory: Path, extensions: List[str]) -> List[Path]:
    """Recursively collect image files from a directory."""
    images = []
    for ext in extensions:
        pattern = f"*{ext}" if not ext.startswith(".") else f"*{ext}"
        # Normalize extension to include dot
        if not ext.startswith("."):
            ext = "." + ext
        images.extend(directory.rglob(pattern))
    return sorted(set(images))


def collect_images_from_list(list_file: Path, extensions: List[str]) -> List[Path]:
    """
    Read a text file containing one image path per line.
    Returns resolved absolute paths that exist and match allowed extensions.
    """
    images = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line).expanduser().resolve()
            if p.is_file():
                images.append(p)
    return sorted(set(images))


def save_captions_to_text(
    directory: Optional[Path],
    results: List[tuple],  # list of (image_path, caption)
    output_filename: str = "captions.txt",
) -> Path:
    """
    Save captions to a text file.
    If directory is provided, save into that directory.
    Otherwise, save into the common parent directory of all images.
    Format per line: <relative_or_absolute_path>\t<caption>
    """
    if not results:
        return None

    if directory is not None:
        out_dir = Path(directory).resolve()
    else:
        # Determine common ancestor
        paths = [Path(r[0]).resolve() for r in results]
        try:
            common = os.path.commonpath([str(p) for p in paths])
            if os.path.isfile(common):
                out_dir = Path(os.path.dirname(common))
            else:
                out_dir = Path(common)
        except ValueError:
            # Fallback to first image's directory
            out_dir = paths[0].parent

    out_path = (out_dir / output_filename).resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        for img_path, caption in results:
            f.write(f"{img_path}\t{caption}\n")
    return out_path


def process_batch(
    images: List[Path],
    api_url: str,
    api_key: str,
    model: str,
    prompt: Optional[str],
    timeout: int,
    output_dir_override: Optional[Path] = None,
) -> Path:
    """
    Process a list of images sequentially and save results.
    Returns the path to the saved text file.
    """
    results = []
    total = len(images)

    for idx, img_path in enumerate(images, start=1):
        print(f"[{idx}/{total}] Processing: {img_path.name}")
        try:
            resp_json = send_caption_request(
                api_url=api_url,
                api_key=api_key,
                image_path=img_path,
                model=model,
                prompt=prompt,
                timeout=timeout,
            )
            caption = extract_caption(resp_json)
            if not caption:
                print("  [WARN] Empty caption received.")
            else:
                preview = caption[:80].replace("\n", " ")
                print(f"  Caption: {preview}{'...' if len(caption) > 80 else ''}")
            results.append((str(img_path), caption))
        except requests.exceptions.HTTPError as e:
            print(f"  [ERROR] HTTP error: {e}")
            results.append((str(img_path), f"[ERROR] HTTP {e}"))
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((str(img_path), f"[ERROR] {e}"))

    out_file = save_captions_to_text(output_dir_override, results)
    if out_file:
        print(f"\nSaved {len(results)} caption(s) to: {out_file}")
    return out_file


def show_menu() -> None:
    """Display the main menu."""
    print("\n" + "=" * 50)
    print("   BATCH IMAGE CAPTIONING TOOL")
    print("=" * 50)
    print("  [1] Process a directory of images")
    print("  [2] Process an image list file (one path per line)")
    print("  [3] Exit")
    print("-" * 50)


def get_config() -> dict:
    """
    Prompt the user for API configuration.
    Returns a dict with keys: api_url, api_key, model, prompt, timeout.
    """
    print("\n--- API Configuration ---")

    default_url = os.environ.get("OPENAI_API_URL", "http://localhost:8000/v1/chat/completions")
    url = input(f"API endpoint URL [{default_url}]: ").strip() or default_url

    default_key = os.environ.get("OPENAI_API_KEY", "")
    key = input(f"API key [{'(set)' if default_key else '(empty)'}]: ").strip() or default_key

    default_model = os.environ.get("CAPTION_MODEL", "gpt-4o-mini")
    model = input(f"Model name [{default_model}]: ").strip() or default_model

    prompt = input("Custom prompt (blank for default): ").strip() or None

    timeout_str = input("Request timeout seconds [120]: ").strip()
    try:
        timeout = int(timeout_str) if timeout_str else 120
    except ValueError:
        timeout = 120

    return {
        "api_url": url,
        "api_key": key,
        "model": model,
        "prompt": prompt,
        "timeout": timeout,
    }


def main() -> None:
    allowed_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]

    while True:
        show_menu()
        choice = input("Select option [1-3]: ").strip()

        if choice == "1":
            dir_str = input("Enter directory path to process: ").strip()
            if not dir_str:
                print("[ERROR] No directory provided.")
                continue
            d = Path(dir_str).expanduser().resolve()
            if not d.is_dir():
                print(f"[ERROR] Not a valid directory: {d}")
                continue

            images = collect_images_from_directory(d, allowed_extensions)
            if not images:
                print("[WARN] No supported image files found in that directory.")
                continue

            print(f"Found {len(images)} image(s).")
            config = get_config()
            process_batch(
                images=images,
                api_url=config["api_url"],
                api_key=config["api_key"],
                model=config["model"],
                prompt=config["prompt"],
                timeout=config["timeout"],
                output_dir_override=d,
            )

        elif choice == "2":
            list_str = input("Enter path to image list file: ").strip()
            if not list_str:
                print("[ERROR] No list file provided.")
                continue
            lf = Path(list_str).expanduser().resolve()
            if not lf.is_file():
                print(f"[ERROR] Not a valid file: {lf}")
                continue

            images = collect_images_from_list(lf, allowed_extensions)
            if not images:
                print("[WARN] No valid image paths found in the list.")
                continue

            print(f"Found {len(images)} image(s).")
            config = get_config()
            # For list-based processing, save to the common parent or cwd
            process_batch(
                images=images,
                api_url=config["api_url"],
                api_key=config["api_key"],
                model=config["model"],
                prompt=config["prompt"],
                timeout=config["timeout"],
                output_dir_override=None,  # auto-detect common parent
            )

        elif choice == "3":
            print("Exiting.")
            break

        else:
            print("[ERROR] Invalid selection. Choose 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(0)
