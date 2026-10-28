"""
peak memory, per stage, on whatever machine a site is using

A site that runs out of memory reports "it died", and that is all anyone
learns: the process is gone, the log ends mid-stage, and the number that would
say how far over the limit it went was never written down. UCMC gets through
`derive` at around 17.5 GB, which is comfortable on a 64 GB node and fatal on a
16 GB laptop -- and the two sites that have not run yet are both larger than
UCMC. So the first thing to add is not a reduction, it is a measurement.

Three things make this harder than calling `max_rss` once:

  * **Peak, not current.** The interesting number is the high-water mark
    inside a stage, which has usually been freed by the time the stage
    returns.
  * **Per stage, not per process.** All ten stages run in one process, so a
    process-lifetime peak attributes everything to whichever stage happened to
    be worst and says nothing about the rest. The fix is a sampling thread:
    `watch_peak` records the maximum RSS seen while its block runs.
  * **Every platform.** Sites run Windows laptops, Linux clusters and Macs.
    `resource.getrusage` does not exist on Windows; `/proc` does not exist on
    macOS; and `ru_maxrss` is kilobytes on Linux and bytes on macOS, which is
    a factor of 1024 waiting to be reported as a finding.

A peak-RSS integer carries no counts, no identifiers and no paths, so it is
safe to put in the bundle -- which is where it goes, as `peak_rss_mb` on the
stage ledger, alongside the seconds already recorded there.
"""

from __future__ import annotations

import contextlib
import platform
import sys
import threading


def current_rss_bytes() -> int | None:
    """
    this process resident set size right now, or None if it cannot be read

    `None` rather than 0 or a raised exception: memory reporting is diagnostic,
    and a stage must not fail because a platform did not expose a counter.
    """
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            return None
    return None


def peak_rss_bytes() -> int | None:
    """
    the high-water mark the OS itself recorded, or None

    Preferred over the sampled figure where it exists, because it cannot miss
    a spike between two samples. It is process-lifetime on every platform,
    which is why `watch_peak` samples as well rather than instead.

      * Windows -- `peak_wset`, from the process memory counters
      * Linux   -- `VmHWM` in /proc/self/status, in kB
      * macOS   -- `ru_maxrss`, which is BYTES here and kilobytes on Linux;
                   the Linux branch above is preferred for exactly that reason
    """
    try:
        import psutil

        info = psutil.Process().memory_info()
        if hasattr(info, "peak_wset"):  # Windows
            return int(info.peak_wset)
    except Exception:
        pass

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass

    try:
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # The units differ by platform and the difference is a factor of 1024.
        # Darwin reports bytes; Linux and the other BSD-derived platforms
        # report kilobytes.
        return int(maxrss) if platform.system() == "Darwin" else int(maxrss) * 1024
    except Exception:
        return None


class _Peak:
    """the result handle `watch_peak` yields, filled in as the block runs"""

    def __init__(self) -> None:
        self.sampled_bytes: int | None = None
        self.process_peak_bytes: int | None = None

    @property
    def megabytes(self) -> float | None:
        """the best figure available, in MB, or None if nothing could be read"""
        candidates = [
            value
            for value in (self.sampled_bytes, self.process_peak_bytes)
            if value is not None
        ]
        return round(max(candidates) / (1024 * 1024), 1) if candidates else None


@contextlib.contextmanager
def watch_peak(interval: float = 0.5):
    """
    record the maximum RSS seen while the block runs

    The sampling thread is a daemon and swallows its own errors, so the worst
    case is a missing number rather than a stage that fails for a diagnostic.
    `interval` trades resolution for overhead: half a second misses nothing
    that matters over a stage measured in minutes, and costs one `psutil` call
    per sample.

    The OS high-water mark is read at both ends and only a RISE across the
    block is attributed here, so a spike the sampler missed still shows up --
    but a peak set by an earlier stage is not re-reported as this one.
    """
    peak = _Peak()
    before = peak_rss_bytes()
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            rss = current_rss_bytes()
            if rss is not None and (
                peak.sampled_bytes is None or rss > peak.sampled_bytes
            ):
                peak.sampled_bytes = rss
            stop.wait(interval)

    thread = threading.Thread(target=sample, name="peak-rss", daemon=True)
    thread.start()
    try:
        yield peak
    finally:
        stop.set()
        thread.join(timeout=interval * 4)
        after = peak_rss_bytes()
        # Only a peak this block is responsible for. `peak_wset` and `VmHWM`
        # never fall, so reporting `after` directly would hand every later
        # stage the largest stage figure.
        if after is not None and (before is None or after > before):
            peak.process_peak_bytes = after
