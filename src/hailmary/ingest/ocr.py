from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

OCR_EXPLANATION = "OCR means reading text from images."
LOW_OCR_CONFIDENCE_THRESHOLD = 0.55
OCR_SUBPROCESS_TIMEOUT_SECONDS = 60


class LocalOcrError(RuntimeError):
    """Local image-based text reading could not read usable text."""


class LocalOcrDependencyError(LocalOcrError):
    """A local OCR executable is missing."""


@dataclass(frozen=True)
class LocalOcrResult:
    text: str
    confidence: float | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.confidence is None:
            return
        if 0.0 <= self.confidence <= 1.0:
            return
        raise ValueError("OCR confidence must be between 0.0 and 1.0.")


class LocalOcrEngine(Protocol):
    def image_to_text(
        self,
        path: Path,
        *,
        page_number: int | None = None,
    ) -> LocalOcrResult:
        """Read text from a standalone image file."""
        ...

    def pdf_page_to_text(self, path: Path, *, page_number: int) -> LocalOcrResult:
        """Render and read text from one 1-based PDF page."""
        ...


class SubprocessLocalOcrEngine:
    """Local OCR engine backed by PATH-discovered Tesseract and Poppler commands."""

    def image_to_text(
        self,
        path: Path,
        *,
        page_number: int | None = None,
    ) -> LocalOcrResult:
        del page_number
        self._tesseract_path()
        return self._tesseract_image(path)

    def pdf_page_to_text(self, path: Path, *, page_number: int) -> LocalOcrResult:
        self._tesseract_path()
        pdftoppm_path = self._pdftoppm_path()
        with tempfile.TemporaryDirectory(prefix="hailmary-ocr-") as temp_dir_name:
            output_prefix = Path(temp_dir_name) / "page"
            try:
                render_result = subprocess.run(
                    [
                        pdftoppm_path,
                        "-f",
                        str(page_number),
                        "-l",
                        str(page_number),
                        "-png",
                        "-r",
                        "300",
                        str(path),
                        str(output_prefix),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=OCR_SUBPROCESS_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                raise LocalOcrError(
                    "Image-based text reading (OCR) took too long while rendering the "
                    "PDF page and was stopped after 60 seconds. "
                    f"{OCR_EXPLANATION}"
                ) from exc
            except OSError as exc:
                raise LocalOcrError(
                    "Could not render the PDF page for image-based text reading (OCR): "
                    f"{exc}. {OCR_EXPLANATION}"
                ) from exc
            if render_result.returncode != 0:
                raise LocalOcrError(
                    "Could not render the PDF page for image-based text reading (OCR): "
                    f"{_subprocess_detail(render_result)} {OCR_EXPLANATION}"
                )
            rendered_images = sorted(Path(temp_dir_name).glob("page-*.png"))
            if not rendered_images:
                raise LocalOcrError(
                    "Could not render the PDF page for image-based text reading (OCR): "
                    f"no page image was created. {OCR_EXPLANATION}"
                )
            return self._tesseract_image(rendered_images[0])

    def _tesseract_image(self, path: Path) -> LocalOcrResult:
        tesseract_path = self._tesseract_path()
        try:
            result = subprocess.run(
                [tesseract_path, str(path), "stdout", "tsv"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=OCR_SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise LocalOcrError(
                "Image-based text reading (OCR) took too long while reading the image "
                f"and was stopped after 60 seconds. {OCR_EXPLANATION}"
            ) from exc
        except OSError as exc:
            raise LocalOcrError(
                "Could not run image-based text reading (OCR): "
                f"{exc}. {OCR_EXPLANATION}"
            ) from exc
        if result.returncode != 0:
            raise LocalOcrError(
                "Could not run image-based text reading (OCR): "
                f"{_subprocess_detail(result)} {OCR_EXPLANATION}"
            )
        return _result_from_tesseract_tsv(result.stdout)

    def _tesseract_path(self) -> str:
        tesseract_path = shutil.which("tesseract")
        if tesseract_path is None:
            raise LocalOcrDependencyError(
                "Image-based text reading (OCR) needs the local `tesseract` command, "
                f"but Hail Mary could not find it on PATH. {OCR_EXPLANATION}"
            )
        return tesseract_path

    def _pdftoppm_path(self) -> str:
        pdftoppm_path = shutil.which("pdftoppm")
        if pdftoppm_path is None:
            raise LocalOcrDependencyError(
                "Image-based text reading (OCR) needs the local `pdftoppm` command "
                "from Poppler to read PDF page images, but Hail Mary could not find it "
                f"on PATH. {OCR_EXPLANATION}"
            )
        return pdftoppm_path


def _result_from_tesseract_tsv(tsv_text: str) -> LocalOcrResult:
    rows = csv.DictReader(tsv_text.splitlines(), delimiter="\t")
    lines: list[str] = []
    line_words: list[str] = []
    current_line_key: tuple[str, str, str] | None = None
    confidences: list[float] = []

    for row in rows:
        word = (row.get("text") or "").strip()
        confidence = _parse_tesseract_confidence(row.get("conf"))
        if confidence is not None:
            confidences.append(confidence)
        if not word:
            continue
        line_key = (
            row.get("block_num") or "",
            row.get("par_num") or "",
            row.get("line_num") or "",
        )
        if current_line_key is not None and line_key != current_line_key:
            _append_line(lines, line_words)
            line_words = []
        current_line_key = line_key
        line_words.append(word)

    _append_line(lines, line_words)
    confidence = _mean(confidences)
    notes = None
    if confidence is not None and confidence < LOW_OCR_CONFIDENCE_THRESHOLD:
        notes = (
            "Image-based text reading (OCR) finished with low confidence. "
            "Review the source image before relying on this text."
        )
    text = "\n".join(lines).strip()
    if not text:
        notes = _join_notes(
            notes,
            f"Image-based text reading (OCR) found no readable text. {OCR_EXPLANATION}",
        )
    return LocalOcrResult(text=text, confidence=confidence, notes=notes)


def _parse_tesseract_confidence(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except ValueError:
        return None
    if confidence < 0:
        return None
    return max(0.0, min(confidence / 100, 1.0))


def _append_line(lines: list[str], words: list[str]) -> None:
    line = " ".join(words).strip()
    if line:
        lines.append(line)


def _mean(values: Iterable[float]) -> float | None:
    values_list = list(values)
    if not values_list:
        return None
    return sum(values_list) / len(values_list)


def _subprocess_detail(result: subprocess.CompletedProcess[str]) -> str:
    detail = result.stderr.strip() or result.stdout.strip()
    if not detail:
        return f"the local command exited with status {result.returncode}."
    return " ".join(detail.split())[:500]


def _join_notes(*notes: str | None) -> str | None:
    present_notes = [note for note in notes if note]
    if not present_notes:
        return None
    return " ".join(present_notes)
