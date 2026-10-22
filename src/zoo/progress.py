from collections import deque
from time import perf_counter


def format_duration(seconds):
    seconds = max(0, round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes:02}m {seconds:02}s"
    if minutes:
        return f"{minutes}m {seconds:02}s"
    return f"{seconds}s"


class StepProgress:
    """Track a recent training-step rate for concise progress reporting."""

    def __init__(self, step=0, window=20):
        self.window = window
        self.samples = deque([(step, perf_counter())], maxlen=window + 1)

    def complete(self, step):
        self.samples.append((step, perf_counter()))

    def status(self, step, total):
        first_step, first_time = self.samples[0]
        last_step, last_time = self.samples[-1]
        elapsed = last_time - first_time
        rate = (last_step - first_step) / elapsed if elapsed else 0.0
        if rate <= 0:
            return "steps/s=n/a eta=n/a"
        return (
            f"steps/s={rate:.2f} "
            f"eta={format_duration((total - step) / rate)}"
        )
