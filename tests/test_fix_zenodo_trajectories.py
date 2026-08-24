"""
Round-trip tests for `scripts/fix_zenodo_trajectories.py`.

The published trajectories cannot be compared against a ground truth, because the
random numbers that produced them are gone. What can be checked is the repair
itself: reproduce both the buggy and the correct stitching of
`simulateTrajectoryFixedJumps` from the same simulated jump times, feed the buggy
result to the repair, and require it to return the correct one exactly.

Run with `pytest tests/`, or directly with `python tests/test_fix_zenodo_trajectories.py`.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fix_zenodo_trajectories import (  # noqa: E402
    repair_chunk,
    segment_length,
    steady_state_rate,
)

NJUMPS = 48
FACTOR = 1.2


def to_time_delay(array):
    """`paramest_nn.quantum_tools.to_time_delay`, copied to keep these tests
    independent of the `qutip` environment."""
    return np.concatenate((np.asarray([array[0]]), np.diff(array)))


def buggy_stitch(segments, njumps):
    """Reproduce what the published `simulateTrajectoryFixedJumps` recorded.

    Args:
        segments (list of np.ndarray): Jump times of each simulated stretch,
            measured from the start of that stretch.
        njumps (int): Number of delays kept.

    Returns:
        np.ndarray: The recorded time delays, of length `njumps`.
    """
    taus_list = []
    tshift = 0.0
    for col in segments:
        times = tshift + np.asarray(col)
        tshift = times[-1]  # the bug: offset by the last jump, not by tf
        taus_list += list(to_time_delay(times))  # the bug: converted per stretch
        if len(taus_list) >= njumps:
            break
    return np.array(taus_list[:njumps])


def correct_stitch(segments, njumps, tf):
    """The same stitching done right, as in the fixed simulator: collect absolute
    jump times across stretches and convert to delays once, at the end.

    Args:
        segments (list of np.ndarray): Jump times of each simulated stretch.
        njumps (int): Number of delays kept.
        tf (float): Length of a stretch.

    Returns:
        np.ndarray: The correct time delays, of length `njumps`.
    """
    jump_times = []
    tshift = 0.0
    for col in segments:
        jump_times += list(tshift + np.asarray(col))
        tshift += tf
        if len(jump_times) >= njumps:
            break
    return to_time_delay(np.array(jump_times))[:njumps]


def simulate_segments(rng, tf, rate, njumps, force_continuations=0):
    """Draw the jump times of as many stretches as the simulator would have run.

    Args:
        rng (np.random.Generator): Source of randomness.
        tf (float): Length of a stretch.
        rate (float): Mean jump rate.
        njumps (int): Number of delays the simulator was asked for.
        force_continuations (int): Minimum number of continuations to provoke, by
            thinning the first stretches until they fall short of `njumps`.

    Returns:
        list of np.ndarray: Jump times per stretch, measured from its start.
    """
    # Split njumps between the stretches that are forced to fall short, so that
    # their running total stays below njumps and a continuation is unavoidable.
    budget = njumps // (force_continuations + 1) if force_continuations else 0

    segments = []
    total = 0
    while total < njumps:
        if len(segments) < force_continuations:
            count = rng.integers(max(1, budget // 2), budget + 1)
        else:
            count = rng.poisson(rate * tf)
        count = max(int(count), 1)
        col = np.sort(rng.uniform(0.0, tf, size=count))
        segments.append(col)
        total += count
    return segments


def _roundtrip(rng, delta, omega, force_continuations):
    """Check one trajectory end to end. Returns the number of stretches used."""
    tf = float(segment_length(np.array(delta), np.array(omega), NJUMPS, 1.0, FACTOR))
    rate = float(steady_state_rate(np.array(delta), np.array(omega), 1.0))

    segments = simulate_segments(rng, tf, rate, NJUMPS, force_continuations)
    recorded = buggy_stitch(segments, NJUMPS)
    truth = correct_stitch(segments, NJUMPS, tf)

    fixed, repaired, suspect = repair_chunk(recorded[None, :], np.array([tf]))

    np.testing.assert_allclose(fixed[0], truth, rtol=1e-9, atol=1e-9)
    expected_repairs = max(len(segments) - 1, 0) if force_continuations else None
    if expected_repairs is not None:
        assert int(repaired[0]) == expected_repairs, (
            f"repaired {int(repaired[0])} delays, expected {expected_repairs}"
        )
    return len(segments)


def test_single_continuation_is_repaired_exactly():
    """The common case: the first stretch falls short and is continued once."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        delta = rng.uniform(0.0, 3.0)
        omega = rng.uniform(0.25, 5.0)
        _roundtrip(rng, delta, omega, force_continuations=1)


def test_two_continuations_are_repaired_exactly():
    """The rare case of two continuations, where the offsets compound."""
    rng = np.random.default_rng(2)
    for _ in range(200):
        delta = rng.uniform(0.0, 3.0)
        omega = rng.uniform(0.25, 5.0)
        _roundtrip(rng, delta, omega, force_continuations=2)


def test_unaffected_trajectories_are_left_alone():
    """A trajectory whose first stretch already held njumps jumps is untouched."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        delta = rng.uniform(0.0, 3.0)
        omega = rng.uniform(0.25, 5.0)
        tf = float(segment_length(np.array(delta), np.array(omega), NJUMPS, 1.0, FACTOR))
        col = np.sort(rng.uniform(0.0, tf, size=NJUMPS + 10))
        recorded = buggy_stitch([col], NJUMPS)
        fixed, repaired, suspect = repair_chunk(recorded[None, :], np.array([tf]))
        assert int(repaired[0]) == 0, "an unaffected trajectory was modified"
        np.testing.assert_array_equal(fixed[0], recorded)


def test_mixed_batch_is_repaired_row_by_row():
    """Affected and unaffected trajectories repaired together in one array."""
    rng = np.random.default_rng(4)
    recorded, truth, tfs, n_continuations = [], [], [], []
    for i in range(300):
        delta, omega = rng.uniform(0.0, 3.0), rng.uniform(0.25, 5.0)
        tf = float(segment_length(np.array(delta), np.array(omega), NJUMPS, 1.0, FACTOR))
        rate = float(steady_state_rate(np.array(delta), np.array(omega), 1.0))
        # 0, 1 or 2 forced continuations; a "0" stretch may still fall short on
        # its own, which is exactly the mix the published data contains.
        segments = simulate_segments(rng, tf, rate, NJUMPS, i % 3)
        recorded.append(buggy_stitch(segments, NJUMPS))
        truth.append(correct_stitch(segments, NJUMPS, tf))
        tfs.append(tf)
        n_continuations.append(len(segments) - 1)

    fixed, repaired, suspect = repair_chunk(np.array(recorded), np.array(tfs))
    np.testing.assert_allclose(fixed, np.array(truth), rtol=1e-9, atol=1e-9)
    # Exactly one delay is repaired per continuation, and none without one.
    np.testing.assert_array_equal(repaired, np.array(n_continuations))
    assert (repaired == 0).any() and (repaired >= 2).any(), "batch was not mixed"


def test_repaired_delays_are_positive_and_within_a_stretch():
    """A repaired delay is a physical waiting time: positive, and no longer than
    the stretch it straddles plus the remainder of the previous one."""
    rng = np.random.default_rng(5)
    for _ in range(200):
        delta, omega = rng.uniform(0.0, 3.0), rng.uniform(0.25, 5.0)
        tf = float(segment_length(np.array(delta), np.array(omega), NJUMPS, 1.0, FACTOR))
        rate = float(steady_state_rate(np.array(delta), np.array(omega), 1.0))
        segments = simulate_segments(rng, tf, rate, NJUMPS, 1)
        recorded = buggy_stitch(segments, NJUMPS)
        fixed, _, _ = repair_chunk(recorded[None, :], np.array([tf]))
        assert (fixed[0] > 0).all(), "repair produced a non-positive delay"
        assert fixed[0].max() <= 2 * tf, "repair produced an impossibly long delay"


def test_repair_is_idempotent():
    """A second run must leave an already repaired trajectory untouched, and say
    so, rather than mistaking its legitimate crossing for a fresh corruption."""
    rng = np.random.default_rng(7)
    recorded, tfs = [], []
    for i in range(300):
        delta, omega = rng.uniform(0.0, 3.0), rng.uniform(0.25, 5.0)
        tf = float(segment_length(np.array(delta), np.array(omega), NJUMPS, 1.0, FACTOR))
        rate = float(steady_state_rate(np.array(delta), np.array(omega), 1.0))
        segments = simulate_segments(rng, tf, rate, NJUMPS, i % 3)
        recorded.append(buggy_stitch(segments, NJUMPS))
        tfs.append(tf)

    tfs = np.array(tfs)
    once, repaired_once, _ = repair_chunk(np.array(recorded), tfs)
    twice, repaired_twice, suspect = repair_chunk(once, tfs)

    np.testing.assert_array_equal(twice, once), "a second run changed the data"
    assert int(repaired_twice.sum()) == 0, "a second run repaired something again"
    # Every trajectory that was repaired is flagged as already repaired instead.
    np.testing.assert_array_equal(suspect, repaired_once > 0)


def test_steady_state_rate_matches_compute_population_ss():
    """The closed form used to rebuild tf is the one already in the package."""
    rng = np.random.default_rng(6)
    delta = rng.uniform(0.0, 3.0, 50)
    omega = rng.uniform(0.25, 5.0, 50)
    # `compute_population_ss` takes (Omega, Delta), in that order.
    expected = 4 * omega**2 / (1**2 + 4 * delta**2 + 8 * omega**2)
    np.testing.assert_allclose(steady_state_rate(delta, omega, 1.0), expected)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall round-trip tests passed")
