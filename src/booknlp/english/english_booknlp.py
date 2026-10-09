import copy
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import spacy
import torch

from booknlp.common.pipelines import SpacyPipeline
from booknlp.english.bert_qa import QuotationAttribution
from booknlp.english.entity_tagger import LitBankEntityTagger
from booknlp.english.gender_inference_model_1 import GenderEM
from booknlp.english.litbank_coref import LitBankCoref
from booknlp.english.litbank_quote import QuoteTagger
from booknlp.english.name_coref import NameCoref
from booknlp.english.output import (
    write_book_html,
    write_book_json,
    write_entities_csv,
    write_quotes_csv,
    write_supersense_csv,
    write_tokens_csv,
)
from booknlp.english.stage_script import write_stage_script


class EnglishBookNLP:
    def __init__(self, model_params):

        with torch.no_grad():
            start_time = time.time()

            print(model_params)

            spacy_model = "en_core_web_sm"
            if "spacy_model" in model_params:
                spacy_model = model_params["spacy_model"]

            spacy_nlp = spacy.load(spacy_model, disable=["ner"])

            valid_keys = set(["entity", "event", "supersense", "quote", "coref"])

            pipes = model_params["pipeline"].split(",")

            self.gender_cats = [
                ["he", "him", "his"],
                ["she", "her"],
                ["they", "them", "their"],
                ["xe", "xem", "xyr", "xir"],
                ["ze", "zem", "zir", "hir"],
            ]

            if "referential_gender_cats" in model_params:
                self.gender_cats = model_params["referential_gender_cats"]

            project_root = Path(__file__).resolve().parents[2]
            modelPath = os.path.join(project_root, "../model")
            if "model_path" in model_params:
                modelPath = model_params["model_path"]

            if not Path(modelPath).is_dir():
                Path(modelPath).mkdir(parents=True, exist_ok=True)

            model_base_url = (
                "http://people.ischool.berkeley.edu/~dbamman/booknlp_models/"
            )

            def download_with_progress(model_name, destination_path):
                url = f"{model_base_url}{model_name}"
                last_reported_percent = -1

                def reporthook(block_count, block_size, total_size):
                    nonlocal last_reported_percent
                    if total_size <= 0:
                        return
                    downloaded = min(block_count * block_size, total_size)
                    percent = int((downloaded * 100) / total_size)
                    if percent == 100 or (
                        percent % 10 == 0 and percent != last_reported_percent
                    ):
                        print(f"{model_name}: {percent}%")
                        last_reported_percent = percent

                urllib.request.urlretrieve(url, destination_path, reporthook=reporthook)

            if model_params["model"] == "big":
                entityName = "entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model"
                corefName = "coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model"
                quoteAttribName = (
                    "speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model"
                )

                self.entityPath = os.path.join(modelPath, entityName)
                if not Path(self.entityPath).is_file():
                    print("downloading %s" % entityName)
                    download_with_progress(entityName, self.entityPath)

                self.coref_model = os.path.join(modelPath, corefName)
                if not Path(self.coref_model).is_file():
                    print("downloading %s" % corefName)
                    download_with_progress(corefName, self.coref_model)

                self.quoteAttribModel = os.path.join(modelPath, quoteAttribName)
                if not Path(self.quoteAttribModel).is_file():
                    print("downloading %s" % quoteAttribName)
                    download_with_progress(quoteAttribName, self.quoteAttribModel)

            elif model_params["model"] == "small":
                entityName = "entities_google_bert_uncased_L-4_H-256_A-4-v1.0.model"
                corefName = "coref_google_bert_uncased_L-2_H-256_A-4-v1.0.model"
                quoteAttribName = (
                    "speaker_google_bert_uncased_L-8_H-256_A-4-v1.0.1.model"
                )

                self.entityPath = os.path.join(modelPath, entityName)
                if not Path(self.entityPath).is_file():
                    print("downloading %s" % entityName)
                    download_with_progress(entityName, self.entityPath)

                self.coref_model = os.path.join(modelPath, corefName)
                if not Path(self.coref_model).is_file():
                    print("downloading %s" % corefName)
                    download_with_progress(corefName, self.coref_model)

                self.quoteAttribModel = os.path.join(modelPath, quoteAttribName)
                if not Path(self.quoteAttribModel).is_file():
                    print("downloading %s" % quoteAttribName)
                    download_with_progress(quoteAttribName, self.quoteAttribModel)

            elif model_params["model"] == "custom":
                self.entityPath = model_params["entity_model_path"]
                self.coref_model = model_params["coref_model_path"]
                self.quoteAttribModel = model_params["quote_attribution_model_path"]

            self.doEntities = self.doCoref = self.doQuoteAttrib = self.doSS = (
                self.doEvent
            ) = False

            for pipe in pipes:
                if pipe not in valid_keys:
                    print("unknown pipe: %s" % pipe)
                    sys.exit(1)
                if pipe == "entity":
                    self.doEntities = True
                elif pipe == "event":
                    self.doEvent = True
                elif pipe == "coref":
                    self.doCoref = True
                elif pipe == "supersense":
                    self.doSS = True
                elif pipe == "quote":
                    self.doQuoteAttrib = True

            tagsetPath = "data/entity_cat.tagset"
            tagsetPath = os.path.join(os.path.dirname(__file__), tagsetPath)

            if "referential_gender_hyperparameterFile" in model_params:
                self.gender_hyperparameterFile = model_params[
                    "referential_gender_hyperparameterFile"
                ]
            else:
                self.gender_hyperparameterFile = os.path.join(
                    os.path.dirname(__file__), "data/gutenberg_prop_gender_terms.txt"
                )

            pronominalCorefOnly = True

            if "pronominalCorefOnly" in model_params:
                pronominalCorefOnly = model_params["pronominalCorefOnly"]

            if not self.doEntities and self.doCoref:
                print("coref requires entity tagging")
                sys.exit(1)

            if not self.doQuoteAttrib and self.doCoref:
                print("coref requires quotation attribution")
                sys.exit(1)
            if not self.doEntities and self.doQuoteAttrib:
                print("quotation attribution requires entity tagging")
                sys.exit(1)

            self.quoteTagger = QuoteTagger()

            if self.doEntities:
                self.entityTagger = LitBankEntityTagger(self.entityPath, tagsetPath)
                aliasPath = os.path.join(os.path.dirname(__file__), "data/aliases.txt")
                self.name_resolver = NameCoref(aliasPath)

            if self.doQuoteAttrib:
                self.quote_attrib = QuotationAttribution(self.quoteAttribModel)

            if self.doCoref:
                self.litbank_coref = LitBankCoref(
                    self.coref_model,
                    self.gender_cats,
                    pronominalCorefOnly=pronominalCorefOnly,
                )

            self.tagger = SpacyPipeline(spacy_nlp)

            print("--- startup: %.3f seconds ---" % (time.time() - start_time))

    def get_syntax(self, tokens, entities, assignments, genders):

        def check_conj(tok, tokens):
            if tok.deprel == "conj" and tok.dephead != tok.token_id:
                # print("found conj", tok.text)
                return tokens[tok.dephead]
            return tok

        def get_head_in_range(start, end, tokens):
            for i in range(start, end + 1):
                if tokens[i].dephead < start or tokens[i].dephead > end:
                    return tokens[i]
            return None

        agents = {}
        patients = {}
        poss = {}
        mods = {}
        prop_mentions = {}
        pron_mentions = {}
        nom_mentions = {}
        keys = Counter()

        toks_by_children = {}
        for tok in tokens:
            if tok.dephead not in toks_by_children:
                toks_by_children[tok.dephead] = {}
            toks_by_children[tok.dephead][tok] = 1

        for idx, (start_token, end_token, cat, phrase) in enumerate(entities):
            ner_prop = cat.split("_")[0]
            ner_type = cat.split("_")[1]

            if ner_type != "PER":
                continue

            coref = assignments[idx]

            keys[coref] += 1
            if coref not in agents:
                agents[coref] = []
                patients[coref] = []
                poss[coref] = []
                mods[coref] = []
                prop_mentions[coref] = Counter()
                pron_mentions[coref] = Counter()
                nom_mentions[coref] = Counter()

            if ner_prop == "PROP":
                prop_mentions[coref][phrase] += 1
            elif ner_prop == "PRON":
                pron_mentions[coref][phrase] += 1
            elif ner_prop == "NOM":
                nom_mentions[coref][phrase] += 1

            tok = get_head_in_range(start_token, end_token, tokens)
            if tok is not None:
                tok = check_conj(tok, tokens)
                head = tokens[tok.dephead]

                # nsubj
                # mod
                if tok.deprel == "nsubj" and head.lemma == "be":
                    for sibling in toks_by_children[head.token_id]:
                        # "he was strong and happy", where happy -> conj -> strong -> attr/acomp -> be
                        sibling_id = sibling.token_id
                        sibling_tok = tokens[sibling_id]
                        if (
                            sibling_tok.deprel == "attr"
                            or sibling_tok.deprel == "acomp"
                        ) and (sibling_tok.pos == "NOUN" or sibling_tok.pos == "ADJ"):
                            mods[coref].append(
                                {"w": sibling_tok.text, "i": sibling_tok.token_id}
                            )

                            if sibling.token_id in toks_by_children:
                                for grandsibling in toks_by_children[sibling.token_id]:
                                    grandsibling_id = grandsibling.token_id
                                    grandsibling_tok = tokens[grandsibling_id]

                                    if grandsibling_tok.deprel == "conj" and (
                                        grandsibling_tok.pos == "NOUN"
                                        or grandsibling_tok.pos == "ADJ"
                                    ):
                                        mods[coref].append(
                                            {
                                                "w": grandsibling_tok.text,
                                                "i": grandsibling_tok.token_id,
                                            }
                                        )

                # ("Bill and Ted ran" conj captured by check_conj above)
                elif tok.deprel == "nsubj" and head.pos == ("VERB"):
                    agents[coref].append({"w": head.text, "i": head.token_id})

                    # "Bill ducked and ran", where ran -> conj -> ducked
                    for sibling in toks_by_children[head.token_id]:
                        sibling_id = sibling.token_id
                        sibling_tok = tokens[sibling_id]
                        if sibling_tok.deprel == "conj" and sibling_tok.pos == "VERB":
                            agents[coref].append(
                                {"w": sibling_tok.text, "i": sibling_tok.token_id}
                            )

                # "Jack was hit by John and William" conj captured by check_conj above
                elif tok.deprel == "pobj" and head.deprel == "agent":
                    # not root
                    if head.dephead != head.token_id:
                        grandparent = tokens[head.dephead]
                        if grandparent.pos.startswith("V"):
                            agents[coref].append(
                                {"w": grandparent.text, "i": grandparent.token_id}
                            )

                # patient ("He loved Bill and Ted" conj captured by check_conj above)
                elif (
                    tok.deprel == "dobj" or tok.deprel == "nsubjpass"
                ) and head.pos == "VERB":
                    patients[coref].append({"w": head.text, "i": head.token_id})

                # poss

                elif tok.deprel == "poss":
                    poss[coref].append({"w": head.text, "i": head.token_id})

                    # "her house and car", where car -> conj -> house
                    for sibling in toks_by_children[head.token_id]:
                        sibling_id = sibling.token_id
                        sibling_tok = tokens[sibling_id]
                        if sibling_tok.deprel == "conj":
                            poss[coref].append(
                                {"w": sibling_tok.text, "i": sibling_tok.token_id}
                            )

        data = {}
        data["characters"] = []

        for coref, total_count in keys.most_common():
            # must observe a character at least *twice*

            if total_count > 1:
                chardata = {}
                chardata["agent"] = agents[coref]
                chardata["patient"] = patients[coref]
                chardata["mod"] = mods[coref]
                chardata["poss"] = poss[coref]
                chardata["id"] = coref
                if coref in genders:
                    chardata["g"] = genders[coref]
                else:
                    chardata["g"] = None
                chardata["count"] = total_count

                mentions = {}

                pnames = []
                for k, v in prop_mentions[coref].most_common():
                    pnames.append({"c": v, "n": k})
                mentions["proper"] = pnames

                nnames = []
                for k, v in nom_mentions[coref].most_common():
                    nnames.append({"c": v, "n": k})
                mentions["common"] = nnames

                prnames = []
                for k, v in pron_mentions[coref].most_common():
                    prnames.append({"c": v, "n": k})
                mentions["pronoun"] = prnames

                chardata["mentions"] = mentions

                data["characters"].append(chardata)

        return data

    def process(self, filename, outFolder, idd):

        with torch.no_grad():
            start_time = time.time()
            originalTime = start_time

            with open(filename) as file:
                data = file.read()

                if len(data) == 0:
                    print("Input file is empty: %s" % filename)
                    return

                try:
                    os.makedirs(outFolder)
                except FileExistsError:
                    pass

                tokens = self.tagger.tag(data)

                print("--- spacy: %.3f seconds ---" % (time.time() - start_time))
                start_time = time.time()

                if self.doEvent or self.doEntities or self.doSS:
                    entity_vals = self.entityTagger.tag(
                        tokens,
                        doEvent=self.doEvent,
                        doEntities=self.doEntities,
                        doSS=self.doSS,
                    )
                    entity_vals["entities"] = sorted(entity_vals["entities"])
                    if self.doSS:
                        write_supersense_csv(
                            outFolder, idd, entity_vals["supersense"]
                        )

                    if self.doEvent:
                        events = entity_vals["events"]
                        for token in tokens:
                            if token.token_id in events:
                                token.event = "EVENT"

                    write_tokens_csv(outFolder, idd, tokens)

                    print("--- entities: %.3f seconds ---" % (time.time() - start_time))
                    start_time = time.time()

                in_quotes = []
                quotes = self.quoteTagger.tag(tokens)
                system_quote_indexes = set()
                for idx, (q_start, q_end) in enumerate(quotes):
                    if tokens[q_start].text == "[" and tokens[q_end].text == "]":
                        system_quote_indexes.add(idx)

                print("--- quotes: %.3f seconds ---" % (time.time() - start_time))
                start_time = time.time()

                if self.doQuoteAttrib:
                    entities = entity_vals["entities"]
                    attributed_quotations = self.quote_attrib.tag(
                        quotes, entities, tokens
                    )

                    print(
                        "--- attribution: %.3f seconds ---" % (time.time() - start_time)
                    )
                    # return time.time() - start_time
                    start_time = time.time()

                if self.doEntities:
                    entities = entity_vals["entities"]

                    in_quotes = []

                    for start, end, cat, text in entities:
                        if tokens[start].inQuote or tokens[end].inQuote:
                            in_quotes.append(1)
                        else:
                            in_quotes.append(0)

                    # Create entity for first-person narrator, if present
                    refs = self.name_resolver.cluster_narrator(
                        entities, in_quotes, tokens
                    )

                    # Cluster non-PER PROP mentions that are identical
                    refs = self.name_resolver.cluster_identical_propers(entities, refs)

                    # Cluster mentions of named people
                    refs = self.name_resolver.cluster_only_nouns(entities, refs, tokens)

                    print(
                        "--- name coref: %.3f seconds ---" % (time.time() - start_time)
                    )

                    start_time = time.time()

                    # Infer referential gender from he/she/they mentions around characters

                    genderEM = GenderEM(
                        tokens=tokens,
                        entities=entities,
                        refs=refs,
                        genders=self.gender_cats,
                        hyperparameterFile=self.gender_hyperparameterFile,
                    )
                    genders = genderEM.tag(entities, tokens, refs)

                assignments = None
                if self.doEntities:
                    assignments = copy.deepcopy(refs)

                if self.doCoref:
                    torch.cuda.empty_cache()
                    assignments = self.litbank_coref.tag(
                        tokens, entities, refs, genders, attributed_quotations, quotes
                    )

                    print("--- coref: %.3f seconds ---" % (time.time() - start_time))
                    start_time = time.time()

                    ent_names = {}
                    for a, e in zip(assignments, entities):
                        if a not in ent_names:
                            ent_names[a] = Counter()
                        ent_names[a][e[3]] += 1

                    # Update gender estimates from coref data
                    genders = genderEM.update_gender_from_coref(
                        genders, entities, assignments
                    )

                    chardata = self.get_syntax(tokens, entities, assignments, genders)
                    write_book_json(outFolder, idd, chardata)

                if self.doEntities:
                    # Write entities and coref
                    write_entities_csv(outFolder, idd, entities, assignments)

                if self.doQuoteAttrib:
                    write_quotes_csv(
                        outFolder,
                        idd,
                        attributed_quotations,
                        quotes,
                        system_quote_indexes,
                        entities,
                        assignments,
                        tokens,
                    )
                    write_stage_script(
                        outFolder,
                        idd,
                        data,
                        quotes,
                        attributed_quotations,
                        system_quote_indexes,
                        entities,
                        assignments,
                        tokens,
                        genders,
                    )

                if self.doQuoteAttrib and self.doCoref:
                    write_book_html(
                        outFolder,
                        idd,
                        chardata,
                        tokens,
                        entities,
                        assignments,
                        quotes,
                        attributed_quotations,
                        system_quote_indexes,
                    )

                print(
                    "--- TOTAL (excl. startup): %.3f seconds ---, %s words"
                    % (time.time() - originalTime, len(tokens))
                )
                return time.time() - originalTime
