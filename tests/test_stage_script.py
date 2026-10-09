from types import SimpleNamespace

from booknlp.english.stage_script import write_stage_script


def test_system_quote_removes_square_brackets(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    tokens = [
        SimpleNamespace(startByte=0, endByte=1),
        SimpleNamespace(startByte=1, endByte=5),
        SimpleNamespace(startByte=5, endByte=6),
    ]

    write_stage_script(
        output_dir,
        "book",
        "[Dash]",
        [(0, 2)],
        [None],
        {0},
        [],
        [],
        tokens,
    )

    assert (output_dir / "book.stage_script.txt").read_text(
        encoding="utf-8"
    ) == "[SYSTEM]\n\nDash\n"
