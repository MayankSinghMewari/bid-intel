from pathlib import Path

from bid_intel.ingest import build_chunks, parse_qa, read_pages

ADDENDUM = Path(__file__).resolve().parent.parent / "data" / "bids" / "JA-207652" / "addendum_1.pdf"


def test_parses_all_39_items_in_order():
    items = parse_qa(read_pages(ADDENDUM))
    assert [i.qnum for i in items] == list(range(1, 40))


def test_answers_are_attached_to_the_right_question():
    items = {i.qnum: i for i in parse_qa(read_pages(ADDENDUM))}
    assert "USB 3.1 is a minimum requirement" in items[1].answer
    assert "Both are required" in items[30].answer
    assert "Laptops" in items[6].answer


def test_no_header_or_footer_leaks_into_answers():
    for it in parse_qa(read_pages(ADDENDUM)):
        assert "Page " not in it.answer and "ADDENDUM No" not in it.answer


def test_qa_chunking_is_one_chunk_per_item():
    items = parse_qa(read_pages(ADDENDUM))
    chunks = build_chunks(items, "qa")
    assert len(chunks) == 39 and all(len(c.qnums) == 1 for c in chunks)


def test_window_chunks_track_question_numbers():
    items = parse_qa(read_pages(ADDENDUM))
    chunks = build_chunks(items, "window:40:20")
    assert len(chunks) > 39
    assert all(c.qnums for c in chunks)
