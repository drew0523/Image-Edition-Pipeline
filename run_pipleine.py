# run_pipeline_single.py
import os
import sys
import subprocess
import argparse

from mask_rgba import compose_result_single  # 단일 이미지 RGBA 합성 모듈
from overlay import run_folder_cpu_cover           # InsightFace 기반 오버레이 모듈

# === (1) ONNX 추론 설정 ===
MODEL = "./onnx_weights/segface_custom_swinv2_base_512_dynamic.onnx"
SAVE_DIR   = "./onnx_outputs/mask_results"
MASK_DIR   = os.path.join(SAVE_DIR, "swinv2_base_512", "onnx_preds")

# === (2) RGBA 출력 디렉토리 ===
OUTPUT_DIR = "./results_rgba"

# === (3) 최종 합성 출력 폴더 ===
FINAL_OUT_DIR = "./final_results"

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
    parser = argparse.ArgumentParser(description="Single image pipeline")
    parser.add_argument("--image1", required=True, help="입력 이미지 경로")
    parser.add_argument("--image2", required=True, help="합성 배경 이미지 경로")
    args = parser.parse_args()

    image_path = args.image1
    image_path2 = args.image2
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base}.png")
    mask_path = os.path.join(MASK_DIR, base + "_pred.png")

    preflight_onnx()     # (1-0) 사전 점검
    run_infer(image_path) # (1) ONNX 추론

    # (2) 마스크 → RGBA 합성
    compose_result_single(
        image_path=image_path,
        mask_path=mask_path,
        output_path=output_path
    )

    # (3) InsightFace 기반 오버레이
    run_folder_cpu_cover(
        input_dir_or_image=image_path2,  # 기준이 되는 이미지
        overlay_path=output_path,       # 합성된 RGBA 결과
        out_dir=FINAL_OUT_DIR
    )

if __name__ == "__main__":
    main()
