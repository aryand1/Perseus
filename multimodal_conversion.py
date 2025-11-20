"""
perseus_multimodal_preprocessing.py

Multimodal ingestion and preprocessing utilities for PERSEUS.

Scope:
    - Accept text, image, and audio inputs.
    - Normalize all modalities into textual units.
    - Apply lightweight preprocessing (normalization and clause segmentation)
      to produce clause-level text suitable for downstream stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import logging
import pathlib
import re


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawInput:
    """
    Container for unprocessed multimodal inputs.

    Attributes
    ----------
    kind
        One of: "text", "image", "audio".
    payload
        Input content. Typical conventions:
            - For text: raw string or path to a text file.
            - For image: path, bytes, or image object (implementation-defined).
            - For audio: path or audio buffer (implementation-defined).
    metadata
        Arbitrary metadata (e.g., source filename, document id, page number).
    """
    kind: str
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    """
    Textual unit produced by multimodal normalization.

    Attributes
    ----------
    text
        Normalized text string associated with a single source element.
    source_id
        Identifier for the origin of this chunk (e.g., filename, page id).
    modality
        Logical modality of origin:
            "text", "image_ocr", "image_caption", "audio", etc.
    """
    text: str
    source_id: Optional[str] = None
    modality: str = "text"


@dataclass(frozen=True)
class Clause:
    """
    Clause-level unit used as input to the downstream text pipeline.

    Each Clause represents an approximately single-verb segment derived from
    a TextChunk after normalization and clause segmentation.

    Attributes
    ----------
    text
        The clause text.
    source_id
        Identifier of the original source (inherited from TextChunk).
    modality
        Modality label (inherited from TextChunk).
    chunk_index
        Index of the parent TextChunk in the sequence provided to preprocessing.
    clause_index
        Index of this clause within its parent TextChunk.
    """
    text: str
    source_id: Optional[str]
    modality: str
    chunk_index: int
    clause_index: int


# ---------------------------------------------------------------------------
# Multimodal normalization (up to plain text)
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}


class MultimodalNormalizer:
    """
    Normalize heterogeneous inputs (text, image, audio) into TextChunk objects.

    Responsibilities
    ----------------
    - Text: basic wrapping and optional file loading.
    - Image: route to OCR and/or visual captioning.
    - Audio: route to automatic speech recognition (ASR).

    Concrete OCR, captioning, and ASR implementations are delegated to
    `_run_ocr`, `_run_captioning`, and `_run_asr`, which should be provided
    by the integrating application.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("perseus.multimodal")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    # ----------------- public API -----------------

    def normalize(self, raw_inputs: Sequence[RawInput]) -> List[TextChunk]:
        """
        Normalize a sequence of RawInput objects into text chunks.

        This is the primary entry point for multimodal handling.

        Parameters
        ----------
        raw_inputs
            Sequence of RawInput instances representing text, image, or audio.

        Returns
        -------
        List[TextChunk]
            Flattened list of textual chunks corresponding to all inputs.
        """
        chunks: List[TextChunk] = []

        for idx, item in enumerate(raw_inputs):
            source_id = item.metadata.get("id") or f"input_{idx}"
            kind = item.kind.lower()

            if kind == "text":
                self.logger.debug("normalize: text[%s]", source_id)
                chunks.append(self._from_text(item.payload, source_id=source_id))

            elif kind == "image":
                self.logger.info("normalize: image[%s]", source_id)
                chunks.extend(self._from_image(item.payload, source_id=source_id))

            elif kind == "audio":
                self.logger.info("normalize: audio[%s]", source_id)
                chunks.extend(self._from_audio(item.payload, source_id=source_id))

            else:
                self.logger.warning(
                    "normalize: unsupported kind '%s' for %s", kind, source_id
                )

        return chunks

    # ----------------- text handling -----------------

    def _from_text(self, payload: Any, source_id: Optional[str]) -> TextChunk:
        """
        Construct a TextChunk from textual input.

        The default behavior interprets the payload as a UTF-8 string. If the
        application uses file paths, additional file loading can be added in a
        wrapper or subclass.
        """
        text = str(payload)
        return TextChunk(text=text, source_id=source_id, modality="text")

    # ----------------- image handling -----------------

    def _from_image(self, payload: Any, source_id: Optional[str]) -> List[TextChunk]:
        """
        Construct one or more TextChunk objects from an image input.

        Typical usage:
        - Apply OCR to recover text from documents, slides, or forms.
        - Apply visual captioning to summarize scene-based images.
        - Optionally emit both OCR and caption text as independent chunks.
        """
        image = payload

        ocr_text = self._run_ocr(image)
        caption_text = self._run_captioning(image)

        chunks: List[TextChunk] = []

        if ocr_text:
            chunks.append(
                TextChunk(
                    text=ocr_text,
                    source_id=source_id,
                    modality="image_ocr",
                )
            )

        if caption_text:
            chunks.append(
                TextChunk(
                    text=caption_text,
                    source_id=source_id,
                    modality="image_caption",
                )
            )

        if not chunks:
            self.logger.warning(
                "image[%s]: OCR and captioning produced no textual output", source_id
            )
            chunks.append(
                TextChunk(
                    text="",
                    source_id=source_id,
                    modality="image_ocr",
                )
            )

        return chunks

    # ----------------- audio handling -----------------

    def _from_audio(self, payload: Any, source_id: Optional[str]) -> List[TextChunk]:
        """
        Construct one or more TextChunk objects from an audio input.

        Typical usage:
        - Decode audio from the payload.
        - Run ASR to obtain a transcript.
        - Optionally segment long recordings into shorter units.
        """
        audio = payload
        transcript = self._run_asr(audio)

        if not transcript:
            self.logger.warning("audio[%s]: ASR produced no transcript", source_id)
            transcript = ""

        return [
            TextChunk(
                text=transcript,
                source_id=source_id,
                modality="audio",
            )
        ]

    # ----------------- extension points -----------------

    def _run_ocr(self, image: Any) -> str:
        """
        Apply optical character recognition (OCR) to an image.

        Returns
        -------
        str
            Recognized text as a single string. Implementations may internally
            perform layout analysis and line grouping.
        """
        raise NotImplementedError("OCR implementation must be provided by the application.")

    def _run_captioning(self, image: Any) -> str:
        """
        Generate a natural language caption for an image.

        Returns
        -------
        str
            Caption text describing the visual content.
        """
        raise NotImplementedError(
            "Image captioning implementation must be provided by the application."
        )

    def _run_asr(self, audio: Any) -> str:
        """
        Transcribe an audio signal into text.

        Returns
        -------
        str
            Transcript of spoken content.
        """
        raise NotImplementedError(
            "ASR implementation must be provided by the application."
        )

    # ----------------- path-based convenience -----------------

    @staticmethod
    def infer_kind_from_path(path: pathlib.Path) -> str:
        """
        Infer input kind from file extension.

        Parameters
        ----------
        path
            Filesystem path.

        Returns
        -------
        str
            One of: "image", "audio", or "text" (default).
        """
        ext = path.suffix.lower()
        if ext in SUPPORTED_IMAGE_EXTS:
            return "image"
        if ext in SUPPORTED_AUDIO_EXTS:
            return "audio"
        return "text"


# ---------------------------------------------------------------------------
# Minimal Triad–style preprocessing
# ---------------------------------------------------------------------------

class MinimalTriadPreprocessor:
    """
    Lightweight implementation of Stage 1 preprocessing.

    Responsibilities
    ----------------
    1. Text normalization:
       - Remove simple boilerplate patterns.
       - Normalize whitespace.
    2. Clause segmentation:
       - Split text into sentence-like segments.
       - Further partition segments into shorter clauses based on simple rules.

    This class is intended as a clear, inspectable baseline. Integrators may
    substitute more advanced parsing and segmentation as needed.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("perseus.preprocess")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    # ----------------- public API -----------------

    def preprocess_chunks(self, chunks: Sequence[TextChunk]) -> List[Clause]:
        """
        Apply normalization and clause segmentation to a list of TextChunks.

        Parameters
        ----------
        chunks
            Sequence of TextChunk instances produced by the multimodal normalizer.

        Returns
        -------
        List[Clause]
            Flat list of Clause objects, with provenance metadata preserved.
        """
        clauses: List[Clause] = []

        for chunk_index, chunk in enumerate(chunks):
            normalized = self._normalize_text(chunk.text)
            raw_clauses = self._segment_into_clauses(normalized)

            for clause_index, c in enumerate(raw_clauses):
                if not c.strip():
                    continue
                clauses.append(
                    Clause(
                        text=c.strip(),
                        source_id=chunk.source_id,
                        modality=chunk.modality,
                        chunk_index=chunk_index,
                        clause_index=clause_index,
                    )
                )

        return clauses

    # ----------------- normalization -----------------

    def _normalize_text(self, text: str) -> str:
        """
        Perform minimal normalization to clean raw text.

        Operations
        ----------
        - Strip leading and trailing whitespace.
        - Remove simple boilerplate markers.
        - Collapse repeated whitespace into single spaces.
        """
        if not text:
            return ""

        patterns_to_drop = [
            r"click here to read more[:!]*",
            r"^✓\s*",
        ]
        cleaned = text

        for pattern in patterns_to_drop:
            cleaned = re.sub(
                pattern,
                "",
                cleaned,
                flags=re.IGNORECASE | re.MULTILINE,
            )

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    # ----------------- clause segmentation -----------------

    def _segment_into_clauses(self, text: str) -> List[str]:
        """
        Segment normalized text into clause-like units.

        Strategy
        --------
        1. Split on sentence boundaries using punctuation markers.
        2. Within each sentence, split on conjunctions such as "and" to obtain
           shorter, clause-like fragments.

        The resulting segments are intended to approximate single-verb clauses
        and provide a cleaner input for later extraction stages.
        """
        if not text:
            return []

        sentence_candidates = re.split(r"(?<=[.!?])\s+", text)

        clauses: List[str] = []
        for sentence in sentence_candidates:
            sentence = sentence.strip()
            if not sentence:
                continue

            sub_clauses = re.split(r"\band\b", sentence, flags=re.IGNORECASE)
            for fragment in sub_clauses:
                fragment = fragment.strip()
                if fragment:
                    clauses.append(fragment)

        return clauses
