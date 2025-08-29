# picture_edit.py (stable)
import os
from io import BytesIO
from typing import Optional
from PIL import Image

from google import genai
from google.genai import types

def get_client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or os.getenv("GOOGLE_GENAI_API_KEY")
    if not key:
        raise ValueError(
            "Gemini API key not found. "
            "Pass --gemini-api-key or set env var GOOGLE_GENAI_API_KEY."
        )
    return genai.Client(api_key=key)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _save_inline_png(data: bytes, out_path: str):
    im = Image.open(BytesIO(data))
    # 안전하게 PNG로 고정 저장
    im = im.convert("RGBA") if im.mode in ("LA", "P") else im
    im.save(out_path)

def run_geminai_edit(
    input_image_path: str,
    prompt: str,
    out_path: Optional[str] = None,
    out_dir: str = "./geminai_results",
    model: str = "gemini-2.5-flash-image-preview",
    api_key: Optional[str] = None,
    fail_if_no_image: bool = True,
) -> str:
    assert os.path.isfile(input_image_path), f"input image not found: {input_image_path}"
    _ensure_dir(out_dir)

    client = get_client(api_key=api_key)

    # 이미지 응답 요청 (mime_type은 지원 안 해서 제거)
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
    )

    image = Image.open(input_image_path)
    strong_prompt = prompt.strip() + "\n\nReturn the edited image as an image output."

    resp = client.models.generate_content(
        model=model,
        contents=[strong_prompt, image],
        config=config,
    )

    # 후보/콘텐츠 없을 때 방어
    if not getattr(resp, "candidates", None):
        msg = "Gemini response has no candidates."
        if fail_if_no_image:
            raise RuntimeError(msg)
        print("[Gemini]", msg, "Returning input image.")
        return input_image_path

    cand = resp.candidates[0]
    parts = getattr(cand, "content", None) and getattr(cand.content, "parts", None)
    if not parts:
        msg = "Gemini response has no content parts."
        if fail_if_no_image:
            raise RuntimeError(msg)
        print("[Gemini]", msg, "Returning input image.")
        return input_image_path

    def _default_out(idx: int) -> str:
        base = os.path.splitext(os.path.basename(input_image_path))[0]
        return os.path.join(out_dir, f"{base}__geminai_{idx}.png")

    saved = None
    texts = []

    for i, part in enumerate(parts):
        if getattr(part, "text", None):
            texts.append(part.text)
            print("[Gemini text]", part.text)
            continue

        if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
            outp = out_path or _default_out(i)
            _save_inline_png(part.inline_data.data, outp)
            print(f"[Gemini image saved: inline] {outp}")
            saved = outp
            continue

        if getattr(part, "file_data", None) and getattr(part.file_data, "file_uri", None):
            blob = client.files.download(part.file_data.file_uri)  # bytes
            outp = out_path or _default_out(i)
            _save_inline_png(blob, outp)
            print(f"[Gemini image saved: file_uri] {outp}")
            saved = outp
            continue

    if not saved:
        if fail_if_no_image:
            msg = "Gemini responded with no image parts."
            if texts:
                msg += " Text:\n" + "\n---\n".join(texts)
            raise RuntimeError(msg)
        else:
            print("[Gemini] No image parts; returning input image.")
            return input_image_path

    return saved
