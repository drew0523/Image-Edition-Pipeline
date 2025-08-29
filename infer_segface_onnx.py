# 단일 이미지 ONNX 추론 스크립트
import os, argparse, numpy as np, onnx, onnxruntime as ort
import cv2

def load_image_for_model(img_path: str, res: int):
    bgr = cv2.imread(img_path); assert bgr is not None, f"Cannot read {img_path}"
    h0, w0 = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_res = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_CUBIC).astype(np.float32)/255.0
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    rgb_res = (rgb_res - mean) / std
    x = np.transpose(rgb_res, (2,0,1))[None, ...].astype(np.float32)  # (1,3,res,res)
    return x, (h0, w0), os.path.basename(img_path)

def load_ort(onnx_path, use_gpu=False):
    so = ort.SessionOptions()
    providers = (["CUDAExecutionProvider","CPUExecutionProvider"]
                 if use_gpu else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name
    return sess, inp, out

def softmax_np(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)

def setup_save_dirs(save_root: str, model_tag: str):
    model_dir = os.path.join(save_root, model_tag)
    onnx_dir  = os.path.join(model_dir, "onnx_preds")
    npy_dir   = os.path.join(model_dir, "npy")
    os.makedirs(onnx_dir, exist_ok=True)
    os.makedirs(npy_dir,  exist_ok=True)
    log_path = os.path.join(model_dir, "onnx_infer_log.txt")
    return model_dir, onnx_dir, npy_dir, log_path

def log_both(line: str, log_path: str):
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + ("\n" if not line.endswith("\n") else ""))

def main():
    ap = argparse.ArgumentParser(description="ONNX 추론 전용 스크립트 (단일 이미지)")
    ap.add_argument("--onnx",  required=True, help="ONNX 모델 경로")
    ap.add_argument("--image", required=True, help="단일 이미지 경로")
    ap.add_argument("--res",   type=int, default=512, help="모델 입력 해상도 (정사각 기준)")
    ap.add_argument("--gpu",   action="store_true", help="가능하면 GPU 실행(CUDAExecutionProvider)")
    ap.add_argument("--softmax", action="store_true", help="출력에 softmax 적용 후 저장(선택)")
    ap.add_argument("--save_logits", action="store_true", help="logits를 .npy로 저장")
    ap.add_argument("--save_probs",  action="store_true", help="softmax 확률을 .npy로 저장 (softmax 필요)")
    ap.add_argument("--save_dir", type=str, default="onnx_infer_results", help="저장 루트")
    ap.add_argument("--model_tag", type=str, default="model", help="저장 하위 폴더명 태그")
    args = ap.parse_args()

    # ONNX 무결성 체크
    print("== ONNX checker ==")
    onnx_model = onnx.load(args.onnx)
    onnx.checker.check_model(onnx_model)
    print("  ✅ onnx.checker passed")

    # 세션 로드
    print("== Load ONNX Runtime ==")
    ort_sess, ort_in, ort_out = load_ort(args.onnx, use_gpu=args.gpu)
    print(f"  Providers: {ort_sess.get_providers()}")

    # 저장 폴더/로그 준비
    model_dir, onnx_dir, npy_dir, log_path = setup_save_dirs(args.save_dir, args.model_tag)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"ONNX: {args.onnx}\nProviders: {ort_sess.get_providers()}\n")
        f.write(f"Res: {args.res}, softmax: {args.softmax}, save_logits: {args.save_logits}, save_probs: {args.save_probs}\n")
        f.write("="*80 + "\n")

    # 단일 이미지만 처리
    img_path = args.image
    x_np, (h0, w0), base = load_image_for_model(img_path, args.res)
    tag = os.path.splitext(base)[0]
    log_both(f"== Image: {img_path} ==", log_path)

    # ONNX 추론
    y_onnx = ort_sess.run([ort_out], {ort_in: x_np})[0]  # (1,C,H,W)
    log_both(f"  onnx out: {y_onnx.shape}", log_path)

    if args.save_logits:
        logits_path = os.path.join(npy_dir, f"{tag}_logits.npy")
        np.save(logits_path, y_onnx)
        log_both(f"  saved logits -> {logits_path}", log_path)

    if args.softmax or args.save_probs:
        probs = softmax_np(y_onnx, axis=1)
        if args.save_probs:
            probs_path = os.path.join(npy_dir, f"{tag}_probs.npy")
            np.save(probs_path, probs)
            log_both(f"  saved probs  -> {probs_path}", log_path)
    else:
        probs = None

    # argmax 예측 마스크 (원본 해상도로 리사이즈)
    pred = (probs if probs is not None else y_onnx).argmax(1).astype(np.uint8)  # (B,H,W)
    pm = pred[0]  # (H,W), uint8
    pm = cv2.resize(pm, (w0, h0), interpolation=cv2.INTER_NEAREST)

    out_png = os.path.join(onnx_dir, f"{tag}_pred.png")
    cv2.imwrite(out_png, (pm * 255).astype(np.uint8))
    log_both(f"  saved mask  -> {out_png}", log_path)

    log_both(f"\nDone. Results & log: {model_dir}", log_path)

if __name__ == "__main__":
    main()
