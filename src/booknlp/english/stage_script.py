import re
from os.path import join


def _speaker_names(entities, assignments):
    names = {}
    for entity, assignment in zip(entities, assignments or []):
        if assignment is None:
            continue
        category = entity[2].split("_", 1)[0]
        priority = {"PROP": 0, "NOM": 1, "PRON": 2}.get(category, 3)
        current = names.setdefault(assignment, [])
        current.append((priority, entity[3]))

    resolved = {}
    for assignment, candidates in names.items():
        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
        resolved[assignment] = candidates[0][1]
    return resolved


def _speaker_gender(assignment, genders):
    gender = (genders or {}).get(assignment)
    if not gender or gender.get("max", 0) <= 0.6:
        return None
    argmax = gender.get("argmax")
    if argmax == "he/him/his":
        return "M"
    if argmax == "she/her":
        return "F"
    return None


def _speaker_label(assignment, names, genders):
    if assignment is None:
        return "UNKNOWN"
    name = names.get(assignment)
    label = f"{name} #{assignment}" if name else f"CHARACTER #{assignment}"
    gender = _speaker_gender(assignment, genders)
    return f"{label} {gender}" if gender else label


def _append_stage_block(blocks, label, text):
    text = text.strip()
    if text:
        blocks.append((label, text))


def _normalize_system_quote(text):
    return text.replace("[", "").replace("]", "")


def _append_narrator_paragraphs(blocks, text):
    for paragraph in re.split(r"\r?\n\s*\r?\n", text):
        _append_stage_block(blocks, "NARRATOR", paragraph)


def write_stage_script(
    out_folder,
    book_id,
    source_text,
    quotes,
    attributed_quotations,
    system_quote_indexes,
    entities,
    assignments,
    tokens,
    genders=None,
):
    """Write source text as ordered, speaker-labelled stage-script blocks."""
    names = _speaker_names(entities, assignments)
    blocks = []
    cursor = 0

    for idx, (quote_start, quote_end) in enumerate(quotes):
        start = tokens[quote_start].startByte
        end = tokens[quote_end].endByte
        _append_narrator_paragraphs(blocks, source_text[cursor:start])

        quote_text = source_text[start:end]
        if idx in system_quote_indexes:
            label = "SYSTEM"
            quote_text = _normalize_system_quote(quote_text)
        else:
            mention = attributed_quotations[idx]
            assignment = assignments[mention] if mention is not None else None
            label = _speaker_label(assignment, names, genders)
        _append_stage_block(blocks, label, quote_text)
        cursor = end

    _append_narrator_paragraphs(blocks, source_text[cursor:])

    with open(
        join(out_folder, f"{book_id}.stage_script.txt"),
        "w",
        encoding="utf-8",
    ) as out:
        for idx, (label, text) in enumerate(blocks):
            if idx:
                out.write("\n\n")
            out.write(f"[{label}]\n\n{text}")
        if blocks:
            out.write("\n")
