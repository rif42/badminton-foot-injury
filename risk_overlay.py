"""On-screen HUD for live injury-risk output."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from injury_risk import RiskResult


# Colors are BGR.
COLOR_OK: Tuple[int, int, int] = (0, 255, 0)
COLOR_WARN: Tuple[int, int, int] = (0, 200, 255)
COLOR_BAD: Tuple[int, int, int] = (0, 0, 255)
COLOR_IDLE: Tuple[int, int, int] = (128, 128, 128)
COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)
COLOR_PANEL: Tuple[int, int, int] = (30, 30, 30)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
LINE_HEIGHT = 24


class RiskOverlay:
    """Draws injury-risk status and per-parameter alerts on an OpenCV frame."""

    def draw(
        self,
        frame: np.ndarray,
        result: Optional[RiskResult],
        profile_label: Optional[str] = None,
    ) -> np.ndarray:
        """Mutates ``frame`` in place and returns it.

        Args:
            frame: OpenCV BGR image to annotate.
            result: Risk evaluation result, or None when pose data is
                insufficient to compute a score.
            profile_label: Optional active profile label to render in the
                top-right panel. When ``result`` is present, its
                ``profile_label`` takes precedence.
        """
        label = result.profile_label if result is not None else profile_label

        if result is None:
            self._draw_panel(frame, 0, 0, 420, 70, COLOR_PANEL)
            self._draw_text(
                frame,
                "Insufficient data -- pose not fully visible",
                15,
                35,
                COLOR_TEXT,
            )
            if label is not None:
                self._draw_profile_panel(frame, label)
            return frame

        color = self._status_color(result.status)
        # Status badge
        self._draw_panel(frame, 0, 0, 340, 110, color)
        self._draw_text(
            frame,
            f"{self._status_prefix(result.status)} {result.status}",
            15,
            35,
            (0, 0, 0),
        )
        self._draw_text(frame, f"Risk: {result.total_risk:.1f}", 15, 80, (0, 0, 0))

        # Profile indicator panel (top-right)
        self._draw_profile_panel(frame, label)

        # Per-parameter alerts
        y_start = 130
        panel_height = len(result.alerts) * LINE_HEIGHT + 30
        self._draw_panel(frame, 0, y_start, 300, panel_height, COLOR_PANEL)
        for i, alert in enumerate(result.alerts):
            y = y_start + 25 + i * LINE_HEIGHT
            self._draw_text(frame, alert["label"], 15, y, COLOR_TEXT, scale=0.5)
            self._draw_text(
                frame, alert["txt"], 180, y, self._level_color(alert["cls"]), scale=0.5
            )

        return frame

    def _draw_profile_panel(self, frame: np.ndarray, label: str) -> None:
        """Draw the active profile label in a dedicated top-right panel."""
        frame_h, frame_w = frame.shape[:2]
        panel_w, panel_h = 250, 45
        panel_x = frame_w - panel_w
        panel_y = 0
        self._draw_panel(frame, panel_x, panel_y, panel_w, panel_h, COLOR_PANEL)
        self._draw_text(
            frame,
            f"Profile: {label}",
            panel_x + 15,
            panel_y + 30,
            COLOR_TEXT,
            scale=0.65,
        )

    def _status_color(self, status: str) -> Tuple[int, int, int]:
        return {
            "Optimal": COLOR_OK,
            "Caution": COLOR_WARN,
            "High Risk": COLOR_BAD,
            "Idle": COLOR_IDLE,
        }.get(status, COLOR_PANEL)

    def _status_prefix(self, status: str) -> str:
        return {
            "Optimal": "OK",
            "Caution": "!",
            "High Risk": "X",
            "Idle": "·",
        }.get(status, "?")

    def _level_color(self, cls: str) -> Tuple[int, int, int]:
        return {
            "ok": COLOR_OK,
            "warn": COLOR_WARN,
            "bad": COLOR_BAD,
        }.get(cls, COLOR_TEXT)

    def _draw_panel(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        color: Tuple[int, int, int],
    ) -> None:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
        scale: float = FONT_SCALE,
    ) -> None:
        cv2.putText(frame, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)
