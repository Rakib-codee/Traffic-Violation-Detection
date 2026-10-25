from scripts.eval_events import _EventWindow, match_events


def ev(type_, start, end):
    return _EventWindow(type_, start, end)


def test_perfect_match_within_tolerance():
    predicted = [ev("wrong_way", 20, 29)]
    expected = [ev("wrong_way", 22, 31)]
    assert match_events(predicted, expected, tolerance=15) == (1, 0, 0)


def test_type_mismatch_is_fp_and_fn():
    predicted = [ev("red_light", 20, 29)]
    expected = [ev("wrong_way", 20, 29)]
    assert match_events(predicted, expected, tolerance=15) == (0, 1, 1)


def test_outside_tolerance_is_fp_and_fn():
    predicted = [ev("wrong_way", 100, 110)]
    expected = [ev("wrong_way", 20, 29)]
    assert match_events(predicted, expected, tolerance=15) == (0, 1, 1)


def test_one_to_one_matching_counts_duplicate_as_fp():
    predicted = [ev("wrong_way", 20, 29), ev("wrong_way", 22, 31)]
    expected = [ev("wrong_way", 20, 29)]
    assert match_events(predicted, expected, tolerance=15) == (1, 1, 0)


def test_empty_inputs():
    assert match_events([], [], tolerance=15) == (0, 0, 0)
    assert match_events([ev("wrong_way", 1, 5)], [], tolerance=15) == (0, 1, 0)
    assert match_events([], [ev("wrong_way", 1, 5)], tolerance=15) == (0, 0, 1)


def test_overlapping_same_type_events_all_match():
    predicted = [ev("wrong_way", 57, 66), ev("wrong_way", 57, 66)]
    expected = [ev("wrong_way", 57, 66), ev("wrong_way", 57, 66)]
    assert match_events(predicted, expected, tolerance=15) == (2, 0, 0)
