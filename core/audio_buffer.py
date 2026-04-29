from __future__ import annotations
import re

class StableTranscriptBuffer:
    """
    Continuously receive Whisper results, only expose the text that has appeared consistently over multiple runs.

    Improvements v2:
    - Fuzzy suffix matching: "five hours" ≈ "five hours a" → do not reset
    - Soft confirm: similar suffixes gradually accumulate instead of resetting to 1
    - Better confirmed protection: only shrink after 2 consecutive divergences
    """

    def __init__(self, confirm_runs: int = 2, diverge_tolerance: float = 0.35,
                 fuzzy_threshold: float = 0.6):
        self._confirmed        = ""
        self._candidate_suffix = ""
        self._candidate_count  = 0.0
        self._confirm_runs     = confirm_runs
        self._diverge_tol      = diverge_tolerance
        self._fuzzy_threshold  = fuzzy_threshold
        self._history: list[str] = []
        self._max_history      = 5
        self._diverge_streak   = 0

    # def _common_prefix_words(self, a: str, b: str) -> int:
    #     wa, wb = a.lower().split(), b.lower().split()
    #     n = 0
    #     for x, y in zip(wa, wb):
    #         if x == y: n += 1
    #         else: break
    #     return n

    # def _prefix_overlap(self, a: str, b: str) -> float:
    #     """Ratio of prefix overlap, relative to the shorter string."""
    #     wa, wb = a.lower().split(), b.lower().split()
    #     if not wa or not wb: return 0.0
    #     common = 0
    #     for x, y in zip(wa, wb):
    #         if x == y: common += 1
    #         else: break
    #     return common / min(len(wa), len(wb))

    def _common_prefix_words(self, a: str, b: str) -> int:
        import re
        wa = [re.sub(r'\[speaker\d+\]|[^\w]', '', w.lower()) for w in a.split()]
        wb = [re.sub(r'\[speaker\d+\]|[^\w]', '', w.lower()) for w in b.split()]
        wa = [w for w in wa if w]
        wb = [w for w in wb if w]
        
        n = 0
        for x, y in zip(wa, wb):
            if x == y: n += 1
            else: break
        return n

    def _prefix_overlap(self, a: str, b: str) -> float:
        import re
        wa = [re.sub(r'\[speaker\d+\]|[^\w]', '', w.lower()) for w in a.split()]
        wb = [re.sub(r'\[speaker\d+\]|[^\w]', '', w.lower()) for w in b.split()]
        wa = [w for w in wa if w]
        wb = [w for w in wb if w]
        
        if not wa or not wb: return 0.0
        n = 0
        for x, y in zip(wa, wb):
            if x == y: n += 1
            else: break
        return n / len(wb)

    @property
    def confirmed(self) -> str:
        return self._confirmed

    def update(self, new_text: str) -> tuple[str, str]:
        """
        Return (confirmed, provisional).
        confirmed  : text stable, displayed clearly
        provisional: suffix not yet stable, displayed faintly
        """
        new_text = new_text.strip()
        if not new_text:
            return self._confirmed, ""

        self._history.append(new_text)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        confirmed_words = self._confirmed.split()
        new_words       = new_text.split()

        # ── 1. Check if confirmed is still valid in new_text ──────────
        if confirmed_words:
            common = self._common_prefix_words(self._confirmed, new_text)
            overlap_ratio = common / len(confirmed_words)

            if overlap_ratio < (1.0 - self._diverge_tol):
                # Diverge — only shrink after 2 consecutive divergences
                self._diverge_streak += 1
                if self._diverge_streak >= 2:
                    safe_words = new_words[:common]
                    self._confirmed        = " ".join(safe_words)
                    self._candidate_suffix = " ".join(new_words[common:])
                    self._candidate_count  = 1.0
                    self._diverge_streak   = 0
                else:
                    # First divergence → keep confirmed, only update provisional
                    self._candidate_suffix = " ".join(new_words[len(confirmed_words):])
                return self._confirmed, self._candidate_suffix
            else:
                self._diverge_streak = 0

        # ── 2. Calculate new suffix (the part after confirmed) ──────────
        suffix_words = new_words[len(confirmed_words):]
        new_suffix   = " ".join(suffix_words).strip()

        # ── 3. Fuzzy match suffix with candidate ──────────────────────
        if self._candidate_suffix:
            exact_match = (new_suffix == self._candidate_suffix)
            prefix_sim  = self._prefix_overlap(new_suffix, self._candidate_suffix)
            fuzzy_match = prefix_sim >= self._fuzzy_threshold

            if exact_match:
                self._candidate_count += 1.0        # exact → +1
            elif fuzzy_match:
                self._candidate_count += 0.6        # similar → +0.6
                self._candidate_suffix = new_suffix  # update with newer suffix
            else:
                # Different → reset but keep a little bit
                self._candidate_suffix = new_suffix
                self._candidate_count  = max(0.3, self._candidate_count * 0.2)
        else:
            self._candidate_suffix = new_suffix
            self._candidate_count  = 1.0

        # ── 4. Promote if enough points ───────────────────────────────────
        if self._candidate_count >= self._confirm_runs and new_suffix:
            self._confirmed        = new_text
            self._candidate_suffix = ""
            self._candidate_count  = 0.0

        return self._confirmed, self._candidate_suffix

    def reset(self):
        self._confirmed        = ""
        self._candidate_suffix = ""
        self._candidate_count  = 0.0
        self._diverge_streak   = 0
        self._history.clear()

# ─────────────────────────────────────────────────────────────────
# Core Pipeline
# ─────────────────────────────────────────────────────────────────
