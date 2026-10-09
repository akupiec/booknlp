import csv
import json
from collections import Counter
from html import escape
from os.path import join


def write_supersense_csv(out_folder, book_id, supersense_entities):
    with open(
        join(out_folder, "%s.supersense.csv" % book_id),
        "w",
        encoding="utf-8",
    ) as out:
        csv_writter = csv.writer(out)
        csv_writter.writerow([
            "start_token", "end_token", "supersense_category", "text"
        ])
        for start, end, cat, text in supersense_entities:
            csv_writter.writerow([start, end, cat, text])


def write_tokens_csv(out_folder, book_id, tokens):
    with open(
        join(out_folder, "%s.tokens.csv" % book_id), "w", encoding="utf-8"
    ) as out:
        csv_writter = csv.writer(out)
        csv_writter.writerow(
            [
                "paragraph_ID",
                "sentence_ID",
                "token_ID_within_sentence",
                "token_ID_within_document",
                "word",
                "lemma",
                "byte_onset",
                "byte_offset",
                "POS_tag",
                "fine_POS_tag",
                "dependency_relation",
                "syntactic_head_ID",
                "event",
            ]
        )

        for token in tokens:
            csv_writter.writerow(
                [
                    token.paragraph_id,
                    token.sentence_id,
                    token.index_within_sentence_idx,
                    token.token_id,
                    token.text,
                    token.lemma,
                    token.startByte,
                    token.endByte,
                    token.pos,
                    token.fine_pos,
                    token.deprel,
                    token.dephead,
                    token.event,
                ]
            )


def write_book_json(out_folder, book_id, character_data):
    with open(
        join(out_folder, "%s.book.json" % book_id), "w", encoding="utf-8"
    ) as out:
        json.dump(character_data, out)


def write_entities_csv(out_folder, book_id, entities, assignments):
    with open(
        join(out_folder, "%s.entities.csv" % book_id),
        "w",
        encoding="utf-8",
    ) as out:
        csv_writter = csv.writer(out)
        csv_writter.writerow(
            ["COREF", "start_token", "end_token", "prop", "cat", "text"]
        )
        for idx, assignment in enumerate(assignments):
            start, end, cat, text = entities[idx]
            ner_prop = cat.split("_")[0]
            ner_type = cat.split("_")[1]
            csv_writter.writerow(
                [assignment, start, end, ner_prop, ner_type, text]
            )


def write_quotes_csv(
    out_folder,
    book_id,
    attributed_quotations,
    quotes,
    system_quote_indexes,
    entities,
    assignments,
    tokens,
):
    with open(
        join(out_folder, "%s.quotes.csv" % book_id), "w", encoding="utf-8"
    ) as out:
        csv_writter = csv.writer(out)
        csv_writter.writerow(
            [
                "quote_start",
                "quote_end",
                "mention_start",
                "mention_end",
                "mention_phrase",
                "char_id",
                "quote",
            ]
        )

        for idx, line in enumerate(attributed_quotations):
            q_start, q_end = quotes[idx]
            mention = line
            if idx in system_quote_indexes:
                e_start = None
                e_end = None
                cat = "SYSTEM"
                speak = "SYSTEM"
            elif mention is not None:
                entity = entities[mention]
                speaker_id = assignments[mention]
                e_start = entity[0]
                e_end = entity[1]
                cat = entity[3]
                speak = speaker_id
            else:
                e_start = None
                e_end = None
                cat = None
                speak = None
            quote = [tok.text for tok in tokens[q_start : q_end + 1]]
            csv_writter.writerow(
                [
                    q_start,
                    q_end,
                    e_start,
                    e_end,
                    cat,
                    speak,
                    " ".join(quote),
                ]
            )


def write_book_html(
    out_folder,
    book_id,
    character_data,
    tokens,
    entities,
    assignments,
    quotes,
    attributed_quotations,
    system_quote_indexes,
):
    names = {}
    for idx, (start, end, cat, text) in enumerate(entities):
        coref = assignments[idx]
        if coref not in names:
            names[coref] = Counter()
        ner_prop = cat.split("_")[0]
        ner_type = cat.split("_")[1]
        if ner_prop == "PROP":
            names[coref][text.lower()] += 10
        elif ner_prop == "NOM":
            names[coref][text.lower()] += 1
        else:
            names[coref][text.lower()] += 0.001

    with open(
        join(out_folder, "%s.book.html" % book_id), "w", encoding="utf-8"
    ) as out:
        out.write("<html>")
        out.write("""<head>
		  <meta charset="UTF-8">
		</head>""")
        out.write("<h2>Named characters</h2>\n")
        for character in character_data["characters"]:
            char_id = character["id"]

            proper_names = character["mentions"]["proper"]
            if len(proper_names) > 0 or char_id == 0:  # 0=narrator
                proper_name_list = "/".join(
                    [
                        "%s (%s)" % (name["n"], name["c"])
                        for name in proper_names
                    ]
                )

                common_names = character["mentions"]["common"]
                common_name_list = "/".join(
                    [
                        "%s (%s)" % (name["n"], name["c"])
                        for name in common_names
                    ]
                )

                char_count = character["count"]

                if char_id == 0:
                    if len(proper_name_list) == 0:
                        proper_name_list = "[NARRATOR]"
                    else:
                        proper_name_list += "/[NARRATOR]"
                out.write(
                    "%s %s %s <br />\n"
                    % (char_count, proper_name_list, common_name_list)
                )

        out.write("<p>\n")

        out.write("<h2>Major entities (proper, common)</h2>")

        major_places = {}
        for prop in ["PROP", "NOM"]:
            major_places[prop] = {}
            for cat in ["FAC", "GPE", "LOC", "PER", "ORG", "VEH"]:
                major_places[prop][cat] = {}

        for idx, (start, end, cat, text) in enumerate(entities):
            coref = assignments[idx]

            ner_prop = cat.split("_")[0]
            ner_type = cat.split("_")[1]
            if ner_prop != "PRON":
                if coref not in major_places[ner_prop][ner_type]:
                    major_places[ner_prop][ner_type][coref] = Counter()
                major_places[ner_prop][ner_type][coref][text] += 1

        max_entities_to_display = 10
        for cat in ["FAC", "GPE", "LOC", "PER", "ORG", "VEH"]:
            out.write("<h3>%s</h3>" % cat)
            for prop in ["PROP", "NOM"]:
                freqs = {}
                for coref in major_places[prop][cat]:
                    freqs[coref] = sum(
                        major_places[prop][cat][coref].values()
                    )

                sorted_freqs = sorted(
                    freqs.items(), key=lambda x: x[1], reverse=True
                )
                for k, v in sorted_freqs[:max_entities_to_display]:
                    ent_names = []
                    for name, count in major_places[prop][cat][
                        k
                    ].most_common():
                        ent_names.append("%s" % (name))
                    out.write("%s %s <br />" % (v, "/".join(ent_names)))
                out.write("<p>")

        out.write("<h2>Text</h2>\n")

        beforeToks = [""] * len(tokens)
        afterToks = [""] * len(tokens)

        lastP = None

        for idx, (start, end, cat, text) in enumerate(entities):
            coref = assignments[idx]
            name = names[coref].most_common(1)[0][0]
            beforeToks[start] += '<font color="#D0D0D0">[</font>'
            afterToks[end] = (
                '<font color="#D0D0D0">]</font><font color="#FF00FF"><sub>%s-%s</sub></font>'
                % (coref, name)
                + afterToks[end]
            )

        for idx, (start, end) in enumerate(quotes):
            mention_id = attributed_quotations[idx]
            if idx in system_quote_indexes:
                speaker_id = "SYSTEM"
                name = "SYSTEM"
            elif mention_id is not None:
                speaker_id = assignments[mention_id]
                name = names[speaker_id].most_common(1)[0][0]
            else:
                speaker_id = "None"
                name = "None"
            beforeToks[start] += '<font color="#666699">'
            afterToks[end] += "</font><sub>[%s-%s]</sub>" % (
                speaker_id,
                name,
            )

        for idx in range(len(tokens)):
            if tokens[idx].paragraph_id != lastP:
                out.write("<p />")
            out.write(
                "%s%s%s "
                % (
                    beforeToks[idx],
                    escape(tokens[idx].text),
                    afterToks[idx],
                )
            )
            lastP = tokens[idx].paragraph_id

        out.write("</html>")
