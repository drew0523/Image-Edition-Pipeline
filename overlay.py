# overlay.py
import os, math
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def is_image(p: str) -> bool:
    return p.lower().endswith(IMAGE_EXT)

def list_images(root: str) -> List[str]:
    files = [os.path.join(root, f) for f in os.listdir(root) if is_image(f)]
    files.sort()
    return files

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ── RetinaFace (InsightFace, CPU 전용) ───────────────────────────────────────
def load_retinaface_cpu(det_size: Tuple[int, int]=(512, 512)):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=det_size)  # CPU only
    return app

# ── 오버레이 IO & 기하 ─────────────────────────────────────────────────────
def read_overlay_rgba(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert im is not None, f"오버레이 이미지를 열 수 없습니다: {path}"
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    elif im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    elif im.shape[2] != 4:
        raise ValueError("지원하지 않는 오버레이 채널 수입니다.")
    return im

def compute_content_bbox_rgba(rgba: np.ndarray) -> Tuple[int,int,int,int,bool]:
    """
    오버레이의 '내용 영역' 바운딩 박스 계산.
    - RGBA: alpha>0인 픽셀
    - RGB:  채널 합>0인 픽셀
    반환: (x0,y0,w,h, found)
    """
    H, W = rgba.shape[:2]
    if rgba.shape[2] == 4:
        mask = rgba[..., 3] > 0
    else:
        mask = np.any(rgba[..., :3] > 0, axis=2)
    if not np.any(mask):
        return 0, 0, W, H, False  # 내용 없음 → 전체 사용 (fallback)
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, (x1 - x0 + 1), (y1 - y0 + 1), True

def resize_rgba_scale(rgba: np.ndarray, scale: float) -> np.ndarray:
    H, W = rgba.shape[:2]
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    interp = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
    return cv2.resize(rgba, (new_w, new_h), interpolation=interp)

def rotate_rgba(rgba: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-3:
        return rgba
    h, w = rgba.shape[:2]
    cx, cy = w/2.0, h/2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    return cv2.warpAffine(
        rgba, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
    )

def alpha_blend_at_topleft(dst_bgr: np.ndarray, overlay_rgba: np.ndarray, topleft_xy: Tuple[int, int]):
    """
    overlay를 좌상단 기준으로 배치하여 알파 블렌딩.
    화면 밖은 자동 크롭.
    """
    H, W = dst_bgr.shape[:2]
    oh, ow = overlay_rgba.shape[:2]
    x1, y1 = map(int, topleft_xy)
    x2, y2 = x1 + ow, y1 + oh

    rx1, ry1 = max(0, x1), max(0, y1)
    rx2, ry2 = min(W, x2), min(H, y2)
    if rx1 >= rx2 or ry1 >= ry2:
        return

    ox1, oy1 = rx1 - x1, ry1 - y1
    ox2, oy2 = ox1 + (rx2 - rx1), oy1 + (ry2 - ry1)

    roi  = dst_bgr[ry1:ry2, rx1:rx2]
    over = overlay_rgba[oy1:oy2, ox1:ox2]

    over_bgr = over[..., :3].astype(np.float32)
    alpha    = (over[..., 3:4].astype(np.float32)) / 255.0
    if np.max(alpha) <= 0:
        return
    base = roi.astype(np.float32)
    comp = alpha * over_bgr + (1 - alpha) * base
    dst_bgr[ry1:ry2, rx1:rx2] = comp.astype(np.uint8)

def eye_angle_deg(kps: np.ndarray) -> float:
    # insightface 5점: [left_eye, right_eye, nose, left_mouth, right_mouth]
    if kps is None or len(kps) < 2:
        return 0.0
    pL, pR = kps[0], kps[1]
    dx, dy = (pR[0] - pL[0]), (pR[1] - pL[1])
    return math.degrees(math.atan2(dy, dx))

def overlay_cover_bbox_by_content(
    dst_bgr: np.ndarray,
    overlay_rgba_src: np.ndarray,
    bbox_xyxy: np.ndarray,         # [x1,y1,x2,y2]
    align_angle_deg: float = 0.0,  # 눈 각도 등으로 회전하고 싶으면 입력
    padding_scale: float = 1.0,    # 1.0보다 크면 살짝 더 크게 덮기
    grow_ratio: float = 0.15,      # bbox 자체를 비율로 확장(가로/세로 동일)
    grow_w_ratio: float = None,    # 가로만 확장
    grow_h_ratio: float = None,    # 세로만 확장
    grow_px: int = 0               # 사방 픽셀 여유
):
    # 원본 bbox 및 중심
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    bcx, bcy = x1 + bw/2.0, y1 + bh/2.0

    # 축별 확장 비율 결정
    gw = grow_w_ratio if grow_w_ratio is not None else grow_ratio
    gh = grow_h_ratio if grow_h_ratio is not None else grow_ratio
    gw = max(0.0, float(gw))
    gh = max(0.0, float(gh))

    # 확장된 bbox 크기(중심 유지)
    bw_exp = int(round(bw * (1.0 + gw))) + 2*max(0, int(grow_px))
    bh_exp = int(round(bh * (1.0 + gh))) + 2*max(0, int(grow_px))
    bw_exp = max(1, bw_exp)
    bh_exp = max(1, bh_exp)

    # (옵션) overlay 회전
    overlay = rotate_rgba(overlay_rgba_src, align_angle_deg) if abs(align_angle_deg) > 1e-3 else overlay_rgba_src.copy()

    # overlay content box
    cx0, cy0, cw, ch, _ = compute_content_bbox_rgba(overlay)

    # 덮기 스케일
    s = padding_scale * max(bw_exp / max(1, cw), bh_exp / max(1, ch))
    overlay_s = resize_rgba_scale(overlay, s)

    # 스케일 후 content 중심
    content_cx_s = (cx0 + cw / 2.0) * s
    content_cy_s = (cy0 + ch / 2.0) * s

    # overlay 좌상단
    top_left_x = int(round(bcx - content_cx_s))
    top_left_y = int(round(bcy - content_cy_s))
    alpha_blend_at_topleft(dst_bgr, overlay_s, (top_left_x, top_left_y))
def _base_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def process_image_cover(
    app,
    img_path: str,
    overlay_rgba: np.ndarray,
    out_path: str = None,                 # 기존 방식 유지(직접 경로 지정)
    *,
    out_dir: str = None,                  # ← 디렉토리만 주고
    overlay_name: str = "overlay.png",    # ← 오버레이 파일 이름(경로) 넣으면
    filename_pattern: str = "{img_base}__over__{ovl_base}_cover.png",  # ← 여기 규칙대로 파일명 생성
    conf_thres: float = 0.5,
    padding_scale: float = 1.0,
    det_align_by_eyes: bool = False,
    draw_debug: bool = False
):
    """
    out_path가 주어지면 그대로 저장.
    out_path가 None이면 out_dir + filename_pattern을 이용해 파일명을 생성해 저장.
    filename_pattern 변수:
      - {img_base}: 입력 이미지 베이스명
      - {ovl_base}: 오버레이 베이스명
    """
    bgr = cv2.imread(img_path)
    assert bgr is not None, f"이미지를 읽을 수 없습니다: {img_path}"
    faces = app.get(bgr)  # CPU

    vis = bgr.copy()
    for f in faces:
        score = float(getattr(f, "det_score", 0.0))
        if score < conf_thres:
            continue
        bbox = f.bbox.astype(int)
        angle = eye_angle_deg(f.kps) if (det_align_by_eyes and hasattr(f, "kps")) else 0.0

        overlay_cover_bbox_by_content(
            vis, overlay_rgba, bbox,
            align_angle_deg=angle,
            padding_scale=padding_scale,
            grow_ratio=0.30,
            grow_px=6
        )

        if draw_debug:
            x1, y1, x2, y2 = bbox.tolist()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
            if hasattr(f, "kps") and f.kps is not None:
                for (px, py) in f.kps.astype(int):
                    cv2.circle(vis, (int(px), int(py)), 2, (0,0,255), -1, lineType=cv2.LINE_AA)

    # === 저장 경로 결정 ===
    if out_path is None:
        assert out_dir is not None, "out_path 또는 out_dir 중 하나는 필요합니다."
        os.makedirs(out_dir, exist_ok=True)
        img_base = _base_name(img_path)
        ovl_base = _base_name(overlay_name)
        out_name = filename_pattern.format(img_base=img_base, ovl_base=ovl_base)
        out_path = os.path.join(out_dir, out_name)
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cv2.imwrite(out_path, vis)
    return out_path

def run_folder_cpu_cover(
    input_dir_or_image: str,
    overlay_path: str,
    out_dir: str = "rf_cover_out_cpu",
    det_size: Tuple[int,int] = (512, 512),
    conf_thres: float = 0.5,
    padding_scale: float = 1.0,
    det_align_by_eyes: bool = False,
    draw_debug: bool = False,
    filename_pattern: str = "{img_base}__over__{ovl_base}_cover.png",
):
    app = load_retinaface_cpu(det_size=det_size)
    overlay_rgba = read_overlay_rgba(overlay_path)
    ensure_dir(out_dir)

    imgs = list_images(input_dir_or_image) if os.path.isdir(input_dir_or_image) else [input_dir_or_image]
    assert imgs, f"입력에 이미지가 없습니다: {input_dir_or_image}"

    outs = []
    for p in tqdm(imgs, desc="processing (CPU cover)"):
        out_path = process_image_cover(
            app, p, overlay_rgba,
            out_path=None,
            out_dir=out_dir,
            overlay_name=overlay_path,       # ← 오버레이 파일명 전달
            filename_pattern=filename_pattern,
            conf_thres=conf_thres,
            padding_scale=padding_scale,
            det_align_by_eyes=det_align_by_eyes,
            draw_debug=draw_debug
        )
        outs.append(out_path)
    print(f"[done] {len(outs)}장 저장 -> {out_dir}")
    return outs
