"""
Review prompts for the automated review panel.

Prompts for each round of the four-round review process:
- Round 1: Independent review
- Round 2: Deep analysis with peer responses
- Round 3: Structured deliberation (minority/majority)
- Round 4: Modification focus for disputed insights
"""

# Round 1
from .round1 import (
    build_round1_prompt,
    parse_round1_response,
)

# Round 2
from .round2 import (
    build_round2_prompt,
    parse_round2_response,
)

# Round 3
from .round3 import (
    build_round3_minority_prompt,
    build_round3_majority_response_prompt,
    build_round3_final_prompt,
    parse_round3_response,
)

# Round 4
from .round4 import (
    build_round4_modification_prompt,
    build_round4_final_vote_prompt,
    parse_round4_modification_response,
    parse_round4_final_vote_response,
)

# Utilities
from .utils import (
    build_round_summary,
)

__all__ = [
    # Round 1
    "build_round1_prompt",
    "parse_round1_response",
    # Round 2
    "build_round2_prompt",
    "parse_round2_response",
    # Round 3
    "build_round3_minority_prompt",
    "build_round3_majority_response_prompt",
    "build_round3_final_prompt",
    "parse_round3_response",
    # Round 4
    "build_round4_modification_prompt",
    "build_round4_final_vote_prompt",
    "parse_round4_modification_response",
    "parse_round4_final_vote_response",
    # Utilities
    "build_round_summary",
]
