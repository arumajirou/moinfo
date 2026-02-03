# /ABS/PATH/split_pdf_pages.py
from __future__ import annotations

from pathlib import Path
import argparse
from pypdf import PdfReader, PdfWriter  # pypdf: PythonのPDF操作ライブラリ


def split_pdf_to_single_pages(pdf_path: str, out_dir: str) -> None:
    """
    入力PDFを「1ページずつ」個別PDFに分割して保存する。
    - 2ページPDFなら2ファイルが出力される
    - それ以外でも全ページ分出力される
    """
    in_path = Path(pdf_path).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists() or in_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"入力PDFが見つからないか、拡張子がpdfではありません: {in_path}")

    reader = PdfReader(str(in_path))
    total = len(reader.pages)

    # 2ページ想定だが、違っても安全に動かす
    if total != 2:
        print(f"注意: 入力PDFのページ数が2ではありません (ページ数={total})。全ページを分割します。")

    stem = in_path.stem
    for i, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)

        one_page_pdf = out_path / f"{stem}_page{i}.pdf"
        with one_page_pdf.open("wb") as f:
            writer.write(f)

    print(f"完了: {total}ページを分割して保存しました -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDFを1ページずつに分割して保存します（2ページPDFにも対応）"
    )
    parser.add_argument(
        "pdf",
        help="入力PDFのフルパス（例: /ABS/PATH/input.pdf）",
    )
    parser.add_argument(
        "out_dir",
        help="出力フォルダのフルパス（例: /ABS/PATH/out）",
    )
    args = parser.parse_args()
    split_pdf_to_single_pages(args.pdf, args.out_dir)


if __name__ == "__main__":
    main()
