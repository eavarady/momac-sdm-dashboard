"""SDM bottleneck detection heuristics.

Renamed from 'bottleneck' to avoid clashing with the optional third-party
package 'bottleneck' that pandas may attempt to import for performance.
"""

from .bottleneck_detector import detect_bottleneck  # noqa: F401

__all__ = ["detect_bottleneck"]
