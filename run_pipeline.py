# run_pipeline_single.py
import os
import sys
import subprocess
import argparse

from mask_rgba import compose_result_single  # 단일 이미지 RGBA 합성 모듈
from overlay import run_folder_cpu_cover           # InsightFace 기반 오버레이 모듈
from picture_edit import run_geminai_edit          # Gemini 호출

# === (1) ONNX 추론 설정 ===
MODEL = "./onnx_weights/segface_custom_swinv2_base_512_dynamic.onnx"
SAVE_DIR   = "./onnx_outputs/mask_results"
MASK_DIR   = os.path.join(SAVE_DIR, "swinv2_base_512", "onnx_preds")

# === (2) RGBA 출력 디렉토리 ===
OUTPUT_DIR = "./results_rgba"

# === (3) 최종 합성 출력 폴더 ===
FINAL_OUT_DIR = "./final_results"

# === (4) Gemini 결과 폴더 ===
GEMINAI_OUT_DIR = "./geminai_results"

# === Gemini 프롬프트 (필요시 수정) ===
# PROMPT = "Please add a small red dot in the top left corner of this image. Return the result strictly as an edited PNG image." (테스트용)
PROMPT = """ 
        "The current image was created by compositing a head or hair from one photo onto another person. "
        "Please refine the image so that the composited head blends naturally with the overall scene, "
        "as if it were part of the original photo. "
        "If a head was overlaid, remove traces of the original hair beneath it and make the visible head "
        "and hairstyle look like they truly belong in the current image. "
        "If the subject has short or bobbed hair but there are long hairs from the background person still visible, "
        "remove the long hair and adjust the image so that only the current short hairstyle remains consistent. "
        "Return the final result strictly as an edited PNG image."
        """

def preflight_onnx():
    print("== ONNX checker ==")
    if not os.path.exists(MODEL):
        print(f"[ERROR] ONNX model not found: {MODEL}")
        sys.exit(1)
    try:
        import onnxruntime as ort
        print("onnxruntime:", ort.__version__, "device:", ort.get_device())
        _ = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
        print("ORT CPU session: OK")
    except Exception as e:
        print("[ERROR] onnxruntime check failed:", e)
        sys.exit(1)

def run_infer(image_path: str):
    cmd = [
        sys.executable, "infer_segface_onnx.py",  # 단일 이미지 infer 스크립트
        "--onnx", MODEL,
        "--image", image_path,
        "--res", "512",
        "--softmax",
        "--save_logits",
        "--save_probs",
        "--save_dir", SAVE_DIR,
        "--model_tag", "swinv2_base_512"
    ]
    print("== Running inference on single image ==")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Single image pipeline with Gemini post-process")
    parser.add_argument("--image1", required=True, help="입력(인물) 이미지 경로")
    parser.add_argument("--image2", required=True, help="합성 배경(또는 기준) 이미지 경로")
    # parser.add_argument("--gemini-prompt", type=str, default=None, help="Gemini에 전달할 프롬프트(선택)")
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash-image-preview", help="Gemini 모델명")
    # parser.add_argument("--gemini-api-key", type=str, default=None, help="직접 API 키 전달(선택, 없으면 환경변수 사용)")
    args = parser.parse_args()

    image_path = args.image1
    image_path2 = args.image2
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base}.png")
    mask_path = os.path.join(MASK_DIR, base + "_pred.png")

    preflight_onnx()      # (1-0) 사전 점검
    run_infer(image_path) # (1) ONNX 추론

    # (2) 마스크 → RGBA 합성
    compose_result_single(
        image_path=image_path,
        mask_path=mask_path,
        output_path=output_path
    )

    # (3) InsightFace 기반 오버레이 (image2에 output_path를 덮어 씌움)
    outs = run_folder_cpu_cover(
        input_dir_or_image=image_path2,   # 기준 이미지(또는 폴더)
        overlay_path=output_path,         # 합성된 RGBA 결과(오버레이)
        out_dir=FINAL_OUT_DIR
    )
    print("overlay outs:", outs[:3])

    # (4) Gemini 후편집(선택) — overlay 결과 중 첫 번째를 Gemini에 투입
    gem_input = outs[0] if len(outs) > 0 else output_path  # 없으면 RGBA 결과 자체를 사용
    print(f"== Gemini post-edit on: {gem_input} ==")
    final_img = run_geminai_edit(
        input_image_path=gem_input,
        prompt=PROMPT,
        out_dir=GEMINAI_OUT_DIR,
        model=args.gemini_model,
        api_key=os.getenv("GOOGLE_GENAI_API_KEY"),
        fail_if_no_image=False,      # 이미지가 안 와도 파이프라인 계속
    )
    print("[Gemini final image]", final_img)


if __name__ == "__main__":
    main()
