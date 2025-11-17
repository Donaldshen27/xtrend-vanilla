"""GP Change-Point Detection for regime segmentation."""

from xtrend.cpd.gp_fitter import GPFitter
from xtrend.cpd.segmenter import GPCPDSegmenter
from xtrend.cpd.types import CPDConfig, RegimeSegment, RegimeSegments

__all__ = [
    "CPDConfig",
    "RegimeSegment",
    "RegimeSegments",
    "GPFitter",
    "GPCPDSegmenter",
]
