"""xtrend.context — Context set construction for few-shot learning."""
from xtrend.context.types import (
    ContextMethod,
    ContextSequence,
    ContextBatch,
)
from xtrend.context.sampler import (
    sample_final_hidden_state,
    sample_time_equivalent,
    sample_cpd_segmented,
)

__all__ = [
    "ContextMethod",
    "ContextSequence",
    "ContextBatch",
    "sample_final_hidden_state",
    "sample_time_equivalent",
    "sample_cpd_segmented",
]
