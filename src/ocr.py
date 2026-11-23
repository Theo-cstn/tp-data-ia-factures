"""Module d'abstraction OCR et implémentation pour LightOnOCR-3 sur Hyperfluid."""

from __future__ import annotations

import base64
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import httpx
import pymupdf

from src.config import HYPERFLUID_API_KEY, OCR_ENDPOINT_URL, OCR_MODEL


class OCRError(RuntimeError):
    """Erreur levée lors de l'exécution d'un appel OCR."""


@dataclass(frozen=True)
class OCRPageResult:
    """Résultat de la reconnaissance OCR d'une page PDF."""

    page_number: int
    text: str
    processing_time_ms: float
    image_width: int
    image_height: int
    image_size_bytes: int
    model: str
    target_longest_edge: int


class OCRProvider(ABC):
    """Interface abstraite permettant d'interchanger de fournisseur d'OCR."""

    @abstractmethod
    def extract_page(
        self,
        pdf_path: Path,
        page_number: int = 0,
        target_longest_edge: int = 1540,
    ) -> OCRPageResult:
        """Extrait le texte d'une page donnée d'un PDF."""

    def extract_document(
        self,
        pdf_path: Path,
        target_longest_edge: int = 1540,
    ) -> list[OCRPageResult]:
        """Extrait le texte de toutes les pages d'un document PDF."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF introuvable : {pdf_path}")

        with pymupdf.open(pdf_path) as doc:
            page_count = len(doc)

        return [
            self.extract_page(
                pdf_path=pdf_path,
                page_number=page_idx,
                target_longest_edge=target_longest_edge,
            )
            for page_idx in range(page_count)
        ]


class LightOnOCRProvider(OCRProvider):
    """Implémentation OCR utilisant LightOnOCR-3 via l'API OpenAI-compatible Hyperfluid."""

    def __init__(
        self,
        endpoint_url: str = OCR_ENDPOINT_URL,
        api_key: str = HYPERFLUID_API_KEY,
        model: str = OCR_MODEL,
        timeout: float = 120.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _render_page_to_png(
        pdf_path: Path,
        page_number: int,
        target_longest_edge: int,
    ) -> tuple[bytes, int, int]:
        """Rend une page PDF en PNG avec le côté le plus long calibré à target_longest_edge."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF introuvable : {pdf_path}")
        if target_longest_edge <= 0:
            raise ValueError("target_longest_edge doit être un entier strictement positif.")

        with pymupdf.open(pdf_path) as doc:
            if page_number < 0 or page_number >= len(doc):
                raise IndexError(
                    f"Numéro de page invalide ({page_number}) pour {pdf_path.name} "
                    f"qui contient {len(doc)} page(s)."
                )
            page = doc[page_number]
            rect = page.rect
            longest_edge = max(rect.width, rect.height)
            if longest_edge <= 0:
                raise ValueError(f"Dimensions de page invalides pour {pdf_path.name} (page {page_number}).")

            zoom = target_longest_edge / longest_edge
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            png_bytes = pixmap.tobytes("png")
            return png_bytes, pixmap.width, pixmap.height

    def _build_payload(self, png_bytes: bytes) -> dict:
        """Construit le payload JSON OpenAI Chat Completions conforme à la model card LightOnOCR-3."""
        b64_image = base64.b64encode(png_bytes).decode("ascii")
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}",
                            },
                        }
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
        }

    @staticmethod
    def _parse_response_text(response_json: dict) -> str:
        """Extrait le texte reconnu depuis la réponse JSON Chat Completions."""
        choices = response_json.get("choices")
        if not choices or not isinstance(choices, list):
            raise OCRError(f"Réponse API invalide (champ 'choices' absent ou vide) : {response_json}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content or "")

    def extract_page(
        self,
        pdf_path: Path,
        page_number: int = 0,
        target_longest_edge: int = 1540,
    ) -> OCRPageResult:
        """Rend la page en PNG, appelle l'endpoint LightOnOCR-3 et mesure la latence."""
        start_time = time.perf_counter()

        png_bytes, img_width, img_height = self._render_page_to_png(
            pdf_path=pdf_path,
            page_number=page_number,
            target_longest_edge=target_longest_edge,
        )
        payload = self._build_payload(png_bytes)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint_url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                response_json = response.json()
        except httpx.HTTPStatusError as exc:
            body_preview = exc.response.text[:500]
            raise OCRError(
                f"Erreur HTTP {exc.response.status_code} depuis l'endpoint OCR : {body_preview}"
            ) from exc
        except httpx.RequestError as exc:
            raise OCRError(f"Erreur réseau lors de l'appel à l'endpoint OCR : {exc}") from exc
        except ValueError as exc:
            raise OCRError(f"Réponse JSON invalide depuis l'endpoint OCR : {exc}") from exc

        text = self._parse_response_text(response_json)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return OCRPageResult(
            page_number=page_number,
            text=text,
            processing_time_ms=elapsed_ms,
            image_width=img_width,
            image_height=img_height,
            image_size_bytes=len(png_bytes),
            model=self.model,
            target_longest_edge=target_longest_edge,
        )
