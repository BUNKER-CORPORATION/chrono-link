from __future__ import annotations

import numpy as np
import pytest

from chrono_link.contracts import FilteredChunk
from chrono_link.windows import WindowAssembler, WindowError


def filtered_chunk(
    start: int,
    samples: int,
    *,
    settled: bool | np.ndarray = True,
) -> FilteredChunk:
    sequence = np.arange(start, start + samples, dtype=np.int64)
    cleaned = np.vstack((sequence, sequence + 10_000)).astype(np.float64)
    flags = (
        np.full(samples, settled, dtype=np.bool_)
        if isinstance(settled, bool)
        else settled
    )
    return FilteredChunk(
        cleaned=cleaned,
        bands={"mi": cleaned + 1, "mu": cleaned + 2},
        timestamps=1_700_000_000.0 + sequence / 250.0,
        sequence=sequence,
        settled=flags,
    )


def test_window_assembler_emits_exact_reference_slices() -> None:
    assembler = WindowAssembler(("C3", "C4"), 250)
    source = filtered_chunk(0, 625)
    windows = assembler.push(source)
    assert [window.start_sequence for window in windows] == [0, 64, 128, 192, 256, 320]
    for window, start in zip(windows, [0, 64, 128, 192, 256, 320], strict=True):
        np.testing.assert_array_equal(window.cleaned, source.cleaned[:, start : start + 250])
        np.testing.assert_array_equal(
            window.bands["mi"], source.bands["mi"][:, start : start + 250]
        )
        assert window.end_sequence == start + 250


def test_window_assembler_handles_arbitrary_chunks_without_duplicates() -> None:
    assembler = WindowAssembler(("C3", "C4"), 250)
    outputs = []
    start = 0
    for size in (1, 249, 10, 300, 65):
        outputs.extend(assembler.push(filtered_chunk(start, size)))
        start += size
    starts = [window.start_sequence for window in outputs]
    assert starts == list(range(0, 321, 64))
    assert len(starts) == len(set(starts))


def test_unsettled_samples_and_reset_suppress_cross_boundary_windows() -> None:
    assembler = WindowAssembler(("C3", "C4"), 250)
    flags = np.arange(1300) >= 1000
    first = assembler.push(filtered_chunk(0, 1300, settled=flags))
    assert [window.start_sequence for window in first] == [1000]

    assembler.reset()
    assert assembler.push(filtered_chunk(1300, 249)) == []
    after = assembler.push(filtered_chunk(1549, 1))
    assert len(after) == 1
    assert after[0].start_sequence == 1300


def test_window_assembler_rejects_invalid_chunks_and_contract_changes() -> None:
    with pytest.raises(ValueError):
        WindowAssembler((), 250)
    with pytest.raises(ValueError):
        WindowAssembler(("C3",), 250, window_samples=0)
    assembler = WindowAssembler(("C3", "C4"), 250)
    empty = filtered_chunk(0, 0)
    assert assembler.push(empty) == []
    with pytest.raises(WindowError, match="channel"):
        assembler.push(
            FilteredChunk(
                cleaned=np.ones((1, 2)),
                bands={"mi": np.ones((1, 2))},
                timestamps=np.arange(2, dtype=np.float64),
                sequence=np.arange(2, dtype=np.int64),
                settled=np.ones(2, dtype=np.bool_),
            )
        )
    nonfinite = filtered_chunk(0, 2)
    nonfinite.cleaned[0, 0] = np.nan
    with pytest.raises(WindowError, match="finite"):
        assembler.push(nonfinite)
    discontinuous = filtered_chunk(0, 3)
    discontinuous.sequence[1] += 3
    with pytest.raises(WindowError, match="contiguous"):
        assembler.push(discontinuous)

    assembler.push(filtered_chunk(0, 10))
    changed = filtered_chunk(10, 10)
    changed = FilteredChunk(
        cleaned=changed.cleaned,
        bands={"beta": changed.cleaned},
        timestamps=changed.timestamps,
        sequence=changed.sequence,
        settled=changed.settled,
    )
    with pytest.raises(WindowError, match="branch"):
        assembler.push(changed)


def test_sequence_gap_defensively_clears_buffer() -> None:
    assembler = WindowAssembler(("C3", "C4"), 250)
    assert assembler.push(filtered_chunk(0, 200)) == []
    assert assembler.push(filtered_chunk(300, 249)) == []
    windows = assembler.push(filtered_chunk(549, 1))
    assert len(windows) == 1
    assert windows[0].start_sequence == 300
