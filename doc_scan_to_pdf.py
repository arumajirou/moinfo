# C:\moinfo\doc_scan_to_pdf_v2.py
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import img2pdf


# ============================================================
# 0) Windows日本語パス対応 I/O
# ============================================================
def imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None

        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img

        pil = Image.open(BytesIO(data.tobytes())).convert("RGB")
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def imwrite_unicode(path: Path, image_bgr: np.ndarray, params: Optional[List[int]] = None) -> bool:
    params = params or []
    ext = path.suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"

    allowed = {".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    if ext not in allowed:
        ext = ".jpg"
        path = path.with_suffix(".jpg")

    ok, buf = cv2.imencode(ext, image_bgr, params)
    if ok:
        buf.tofile(str(path))
        return True

    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(str(path))
        return True
    except Exception:
        return False


# ============================================================
# 1) 幾何：四隅整列・透視変換
# ============================================================
def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def four_point_warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = max(1, int(max(width_a, width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = max(1, int(max(height_a, height_b)))

    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_w, max_h))


def ensure_landscape(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h > w:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


# ============================================================
# 2) 画像前処理（枠検出を安定化）
# ============================================================
def enhance_for_detection(image_bgr: np.ndarray) -> np.ndarray:
    # 明るさムラ耐性のため L成分にCLAHE
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def resize_keep_ratio(img: np.ndarray, max_height: int = 1200) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if h <= max_height:
        return img, 1.0
    ratio = max_height / float(h)
    out = cv2.resize(img, (int(w * ratio), max_height), interpolation=cv2.INTER_AREA)
    return out, ratio


# ============================================================
# 3) 枠検出（2系統）→最大四角形を採用
# ============================================================
def find_quad_via_edges(img_bgr: np.ndarray, canny1: int, canny2: int, eps: float) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, canny1, canny2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours[:30]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def find_quad_via_threshold(img_bgr: np.ndarray, eps: float) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 背景差がある前提で適応的二値化
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours[:30]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def pick_best_quad(img_small: np.ndarray, canny1: int, canny2: int, eps: float) -> Optional[np.ndarray]:
    # 2系統で取れたら、面積が大きい方を採用
    q1 = find_quad_via_edges(img_small, canny1, canny2, eps)
    q2 = find_quad_via_threshold(img_small, eps)

    candidates = []
    if q1 is not None:
        candidates.append(q1)
    if q2 is not None:
        candidates.append(q2)
    if not candidates:
        return None

    best = None
    best_area = -1.0
    for q in candidates:
        area = cv2.contourArea(q.astype(np.float32))
        if area > best_area:
            best_area = area
            best = q
    return best


# ============================================================
# 4) タイトクロップ（ワープ後の余白を削る）
# ============================================================
def tight_crop(img_bgr: np.ndarray, pad: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 「ほぼ白」背景を想定して非白領域を抽出（カード/書類の外側余白を落とす）
    # しきい値はゆるめ（背景が完全白じゃない場合もある）
    _, th = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    H, W = img_bgr.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return img_bgr[y0:y1, x0:x1]


# ============================================================
# 5) 保険証マスキング（ファイル名に「保険」を含む場合のみ）
#    例PDFの①②③④を意識（記号/番号/保険者番号/2次元バーコード）:contentReference[oaicite:2]{index=2}
# ============================================================
def mask_rect(img: np.ndarray, rect: Tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = rect
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)


def detect_qr_and_mask(img_bgr: np.ndarray, margin: int = 10) -> bool:
    det = cv2.QRCodeDetector()
    ok, points = det.detect(img_bgr)
    if not ok or points is None:
        return False

    pts = points.reshape(-1, 2)
    x0 = int(np.min(pts[:, 0])) - margin
    y0 = int(np.min(pts[:, 1])) - margin
    x1 = int(np.max(pts[:, 0])) + margin
    y1 = int(np.max(pts[:, 1])) + margin
    mask_rect(img_bgr, (x0, y0, x1, y1))
    return True


def mask_insurance_fields(img_bgr: np.ndarray, debug_overlay: bool = False) -> np.ndarray:
    """
    OCR無しでやるため「レイアウト比率ベース + テキスト塊検出の補助」でマスクする。
    成功率を上げるため、まず標準サイズへ正規化してから処理する。
    """
    out = img_bgr.copy()

    # 標準化（横長前提）
    out = ensure_landscape(out)
    H, W = out.shape[:2]
    target_w = 1400
    scale = target_w / float(W)
    out = cv2.resize(out, (target_w, int(H * scale)), interpolation=cv2.INTER_AREA)
    H, W = out.shape[:2]

    # (4) 2次元コード：QR検出できたらそこを優先マスク
    _ = detect_qr_and_mask(out, margin=12)

    # (1)(2) 記号・番号：上部の帯（だいたい上 10〜26%）
    # 例PDFでは上部に長い黒塗りが入る領域 :contentReference[oaicite:3]{index=3}
    band1 = (int(W * 0.16), int(H * 0.08), int(W * 0.96), int(H * 0.26))

    # (3) 保険者番号：左下寄りの行（だいたい中下 55〜74%）
    band2 = (int(W * 0.05), int(H * 0.54), int(W * 0.62), int(H * 0.74))

    # バンド内の「文字の塊」を拾って、その外接矩形を少し膨らませてマスク
    def mask_text_blocks_in_roi(roi: Tuple[int, int, int, int], expand: int = 10) -> None:
        x0, y0, x1, y1 = roi
        sub = out[y0:y1, x0:x1].copy()
        g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)

        # 黒文字抽出（黒成分に寄せる）
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7)))
        _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)), iterations=2)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # 取れなければROIをそのままマスク（安全側）
            mask_rect(out, (x0, y0, x1, y1))
            return

        # 横長の塊（番号列）だけ優先しつつ、最終的にunionでマスク
        xs, ys, xe, ye = W, H, 0, 0
        used = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < 200:  # ノイズ除外
                continue
            aspect = w / float(max(1, h))
            if aspect < 2.0 and area < 1500:
                continue
            xs = min(xs, x)
            ys = min(ys, y)
            xe = max(xe, x + w)
            ye = max(ye, y + h)
            used += 1

        if used == 0:
            mask_rect(out, (x0, y0, x1, y1))
            return

        mask_rect(out, (x0 + xs - expand, y0 + ys - expand, x0 + xe + expand, y0 + ye + expand))

        if debug_overlay:
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 3)

    mask_text_blocks_in_roi(band1, expand=14)
    mask_text_blocks_in_roi(band2, expand=12)

    return out


# ============================================================
# 6) メイン：スキャン + 条件マスク + PDF化
# ============================================================
def collect_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(input_path.glob("*")) if p.is_file() and p.suffix.lower() in exts]


def images_to_pdf(image_paths: List[Path], pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert([str(p) for p in image_paths]))


def scan_one(
    in_path: Path,
    out_dir: Path,
    resize_height: int,
    canny1: int,
    canny2: int,
    eps: float,
    tight_crop_pad: int,
    mask_when_contains: str,
    debug: bool,
) -> Path:
    img = imread_unicode(in_path)
    if img is None:
        raise ValueError(f"画像を読み込めませんでした（パス/形式/破損の可能性）: {in_path}")

    original = img.copy()
    small, ratio = resize_keep_ratio(enhance_for_detection(img), max_height=resize_height)

    quad = pick_best_quad(small, canny1=canny1, canny2=canny2, eps=eps)

    # 枠が取れない場合は「回転矩形」で最低限の補正（それでもダメなら全体）
    if quad is None:
        # threshold由来で輪郭を取って最大のminAreaRect
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.float32)
            quad = box
        else:
            quad = np.array([[0, 0], [small.shape[1]-1, 0], [small.shape[1]-1, small.shape[0]-1], [0, small.shape[0]-1]], dtype=np.float32)

    quad_full = (quad / ratio).astype(np.float32)
    warped = four_point_warp(original, quad_full)
    warped = ensure_landscape(warped)
    warped = tight_crop(warped, pad=tight_crop_pad)

    out_dir.mkdir(parents=True, exist_ok=True)

    scanned_path = out_dir / f"{in_path.stem}_scanned.jpg"
    if not imwrite_unicode(scanned_path, warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise ValueError(f"スキャン画像を書き込めませんでした: {scanned_path}")

    final_path = scanned_path

    # 条件マスキング：ファイル名に指定文字列を含むときだけ
    if mask_when_contains and (mask_when_contains in in_path.name):
        masked = mask_insurance_fields(warped, debug_overlay=debug)
        masked_path = out_dir / f"{in_path.stem}_masked.jpg"
        if not imwrite_unicode(masked_path, masked, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise ValueError(f"マスク画像を書き込めませんでした: {masked_path}")
        final_path = masked_path

    if debug:
        # 枠検出の可視化
        dbg = original.copy()
        q = quad_full.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(dbg, [q], isClosed=True, color=(0, 255, 0), thickness=8)
        dbg_path = out_dir / f"{in_path.stem}_debug_quad.jpg"
        _ = imwrite_unicode(dbg_path, dbg, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # ワープ結果も別名で保存（比較用）
        warp_path = out_dir / f"{in_path.stem}_debug_warp.jpg"
        _ = imwrite_unicode(warp_path, warped, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="枠検知→透視補正→タイトクロップ→(条件)保険証マスク→PDF化（日本語パス対応）"
    )
    parser.add_argument("input", help="入力画像ファイル or フォルダ（フルパス推奨）")
    parser.add_argument("--out_dir", required=True, help="出力フォルダ（フルパス推奨）")
    parser.add_argument("--pdf", required=True, help="出力PDFパス（フルパス推奨）")
    parser.add_argument("--no_pdf", action="store_true", help="PDF化せず画像出力のみ")
    parser.add_argument("--debug", action="store_true", help="デバッグ画像出力（枠/ワープ等）")

    parser.add_argument("--resize_height", type=int, default=1200, help="検出用縮小の高さ(px)")
    parser.add_argument("--canny1", type=int, default=60, help="Cannyしきい値1")
    parser.add_argument("--canny2", type=int, default=160, help="Cannyしきい値2")
    parser.add_argument("--eps", type=float, default=0.02, help="四角形近似の許容比率")
    parser.add_argument("--tight_crop_pad", type=int, default=10, help="タイトクロップ時の余白(px)")

    parser.add_argument(
        "--mask_when_contains",
        default="保険",
        help="ファイル名にこの文字列が含まれる時だけマスキング（例：保険 / 保健 など）",
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    pdf_path = Path(args.pdf).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"入力が存在しません: {input_path}")

    images = collect_images(input_path)
    if not images:
        raise FileNotFoundError(f"処理対象画像が見つかりません: {input_path}")

    outputs: List[Path] = []
    for p in images:
        out_img = scan_one(
            in_path=p,
            out_dir=out_dir,
            resize_height=args.resize_height,
            canny1=args.canny1,
            canny2=args.canny2,
            eps=args.eps,
            tight_crop_pad=args.tight_crop_pad,
            mask_when_contains=args.mask_when_contains,
            debug=args.debug,
        )
        outputs.append(out_img)

    if not args.no_pdf:
        images_to_pdf(outputs, pdf_path)
        print(f"完了: {len(outputs)}枚 → PDF作成: {pdf_path}")
    else:
        print(f"完了: {len(outputs)}枚 → 画像出力のみ: {out_dir}")


if __name__ == "__main__":
    main()
