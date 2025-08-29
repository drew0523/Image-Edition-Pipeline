# mask_rgba.py
import os
import cv2

def compose_result_single(
    image_path: str,
    mask_path: str,
    output_path: str,
) -> None:
    """
    원본 이미지(image_path)와 마스크(mask_path)를 매칭해
    RGBA(PNG)로 저장. 마스크를 알파 채널로 사용 (255=불투명).
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"mask not found: {mask_path}")

    img  = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        raise RuntimeError(f"failed to read img/mask: {image_path}, {mask_path}")

    # 크기 맞추기
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 이진화 → 알파
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, mask_bin])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, rgba)
    print(f"[save] {output_path}")

if __name__ == "__main__":
    # 단독 실행용 CLI
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="원본 이미지 경로")
    ap.add_argument("--mask",  required=True, help="마스크 이미지 경로")
    ap.add_argument("--output", required=True, help="출력 PNG 경로")
    args = ap.parse_args()
    compose_result_single(args.image, args.mask, args.output)
