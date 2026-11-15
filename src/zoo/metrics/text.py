"""Language-model metrics."""

from __future__ import annotations

import math


def bits_per_byte(loss_nats: float, num_tokens: int, num_bytes: int) -> float:
    """Bits per UTF-8 byte of text from a mean per-token loss in nats: `loss / ln 2 * tokens / bytes`.

    Comparable across tokenizers. `num_tokens` must count exactly the tokens the loss was averaged over."""
    return loss_nats / math.log(2) * num_tokens / num_bytes
