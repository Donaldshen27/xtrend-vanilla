"""GP Change-Point Detection for regime segmentation."""

from xtrend.cpd.gp_fitter import GPFitter
from xtrend.cpd.segmenter import GPCPDSegmenter
from xtrend.cpd.types import CPDConfig, RegimeSegment, RegimeSegments
from xtrend.cpd.validation import ValidationCheck, ValidationReport

__all__ = [
    "CPDConfig",
    "RegimeSegment",
    "RegimeSegments",
    "GPFitter",
    "GPCPDSegmenter",
    "ValidationCheck",
    "ValidationReport",
]
