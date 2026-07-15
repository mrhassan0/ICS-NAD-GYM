"""Quarantine state machine for the BLOCK action.

BLOCK on a flow's source starts a W-tick quarantine for that source. While quarantined,
subsequent flows from the same source are still observed by the agent (for monitoring
realism) but are auto-resolved: the agent's action that step doesn't go through the normal
reward table, it accrues the quarantine's ongoing bonus/penalty instead. This statefulness
is what makes the environment genuinely sequential rather than a per-flow classifier.
"""


class QuarantineState:
    def __init__(self, duration: int = 50):
        self.duration = duration
        self._remaining: dict[str, int] = {}
        self._was_correct: dict[str, bool] = {}  # True = quarantined an actual attacker

    def reset(self):
        self._remaining.clear()
        self._was_correct.clear()

    def is_quarantined(self, source: str) -> bool:
        return self._remaining.get(source, 0) > 0

    def was_correct_block(self, source: str) -> bool:
        return self._was_correct.get(source, False)

    def start(self, source: str, was_correct: bool):
        self._remaining[source] = self.duration
        self._was_correct[source] = was_correct

    def tick(self):
        """Advance all active quarantines by one step; drop expired entries."""
        expired = []
        for src in self._remaining:
            self._remaining[src] -= 1
            if self._remaining[src] <= 0:
                expired.append(src)
        for src in expired:
            del self._remaining[src]
            del self._was_correct[src]
