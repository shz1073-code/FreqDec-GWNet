"""R2 weak-supervision state-label generation pipeline.

Implements PROJECT_CONSTRAINTS_19.6.md §1 R2 mandates:
    1. full-sequence state label
    2. multi-anchor quality fusion
    3. band-limited filtering
    4. quality_report / valid_mask

The pipeline operates on full bursts (typically 100-300 frames) — *not*
sliding windows — because reliable Hilbert phase estimation needs at least
three respiratory cycles, and multi-anchor consensus needs spatial diversity
across the entire sequence.

Public surface (top-level imports):
    estimate_background_translation     # B.1
    cumulative_position
    band_pass_filter                    # B.2
    project_to_principal_axis
    hilbert_phase_amplitude
    multi_region_consensus              # B.3
    StateLabel                          # B.4 dataclass
    generate_state_label                # B.4 orchestrator
"""

from .background_motion import (
    cumulative_position,
    estimate_background_translation,
)
from .band_filter import (
    band_pass_filter,
    hilbert_phase_amplitude,
    project_to_principal_axis,
)
from .anchor_fusion import multi_region_consensus
from .pipeline import StateLabel, generate_state_label

__all__ = [
    "StateLabel",
    "band_pass_filter",
    "cumulative_position",
    "estimate_background_translation",
    "generate_state_label",
    "hilbert_phase_amplitude",
    "multi_region_consensus",
    "project_to_principal_axis",
]
