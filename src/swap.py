"""Gender counterfactual swapper for English biographies.

Given a spaCy Doc of a bio and the gender of the person it describes, `swap_doc` returns
  * template_orig : the bio with every mention of the subject's name replaced by "{NAME}"
  * template_swap : the same bio with the words that encode the subject's gender (and their
                    relatives' gender) flipped
plus audit fields used to drop bios whose swap cannot be trusted.

Only the *person's* gender changes. Words that describe the job's content ("women's
health", "female patients", "maternity ward", "pregnancy care") and words inside
organisation / place names ("Brigham and Women's Hospital", "King's College") are left alone.

The Doc must come from a pipeline with a tagger (for "her" -> "his"/"him") and NER
(en_core_web_lg with the parser disabled is recommended; en_core_web_sm works but misses
more names). Run `normalize_text` on the raw bio before parsing.
"""
import re
from collections import Counter

# Male -> female, lower-case. "his" and "her" need POS tags; titles need context.
# Both are handled in _swap_word.
M2F = {
    "he": "she", "him": "her", "his": "her", "himself": "herself",
    "mr": "ms", "sir": "madam",
    "man": "woman", "boy": "girl", "gentleman": "lady",
    "husband": "wife", "husbands": "wives", "boyfriend": "girlfriend", "fiance": "fiancee",
    "groom": "bride", "widower": "widow",
    "father": "mother", "fathers": "mothers", "dad": "mom", "daddy": "mommy",
    "son": "daughter", "sons": "daughters", "brother": "sister", "brothers": "sisters",
    "uncle": "aunt", "uncles": "aunts", "nephew": "niece", "nephews": "nieces",
    "grandfather": "grandmother", "grandson": "granddaughter", "grandsons": "granddaughters",
    "stepfather": "stepmother", "stepson": "stepdaughter", "godfather": "godmother",
    "fatherhood": "motherhood", "brotherhood": "sisterhood", "manhood": "womanhood",
    "boyhood": "girlhood", "fraternity": "sorority",
    "actor": "actress", "actors": "actresses", "waiter": "waitress",
    "chairman": "chairwoman", "spokesman": "spokeswoman", "businessman": "businesswoman",
    "sportsman": "sportswoman", "king": "queen", "prince": "princess",
}

# Female -> male: the inverse map, minus "her" (ambiguous), plus one-way extras.
F2M = {f: m for m, f in M2F.items() if f != "her"}
F2M.update({"hers": "his", "mrs": "mr", "mum": "dad", "mommy": "daddy", "ma'am": "sir"})

# Every listed word is flipped, whichever person it refers to, so "his wife" becomes
# "her husband" (relations stay consistent).
SWAP = {**M2F, **F2M}
assert not set(M2F) & set(F2M), "a word cannot be both male and female"

# Courtesy titles: swapped only as "Mr"/"Ms."/"Mrs" before a name, never "MS" (degree, state).
TITLES = {"mr", "ms", "mrs", "miss"}

# Informal gendered words with no safe swap. If one survives, the bio is flagged and dropped.
AUDIT_EXTRA = {"guy", "guys", "lad", "lads", "mister", "grandpa", "papa", "fella", "manly",
               "gal", "gals", "grandma", "mama", "maiden", "girlish", "motherly"}
GENDERED_WORDS = set(SWAP) | {"her"} | AUDIT_EXTRA

# kinship / role words: a PERSON right after one ("her husband, Mark") is not the subject
RELATIONAL = set(SWAP) - {"he", "him", "his", "himself", "she", "hers", "herself",
                          "sir", "madam", "ma'am"} - TITLES

# Gendered words inside these entities are part of a name and are never swapped.
PROTECTED_ENTS = {"ORG", "FAC", "GPE", "LOC", "WORK_OF_ART", "EVENT", "PRODUCT", "LAW"}

# after these, a "his" is an independent possessive ("the decision was his.") -> "hers"
_HIS_INDEPENDENT_NEXT = {"PUNCT", "CCONJ", "SCONJ", "ADP", "AUX", "VERB", "PART"}

_SPLIT_RE = re.compile(r"[^a-z']+")

# ---------------------------------------------------------------- text clean-up
_MOJIBAKE = {"\x92": "'", "‘": "'", "’": "'", "`": "'", "抗": "'s"}  # 抯 = mis-decoded 's
_GLUED = re.compile(r"(?<=[A-Za-z0-9)\]])([.,;:])(?=(?:he|she|his|her|hers|him|himself|herself|mr|mrs|ms|dr)\b)",
                    re.IGNORECASE)
_JUNK = re.compile(r"\s*\|\s*PowerPoint PPT presentation\s*\|\s*free to view\s*", re.IGNORECASE)


def normalize_text(text):
    """Fix scraping artefacts that break tokenisation: mis-encoded apostrophes, sentences
    glued together ("in 2012.he joined"), and slide-site boilerplate."""
    for bad, good in _MOJIBAKE.items():
        text = text.replace(bad, good)
    text = _JUNK.sub(" ", text)
    text = _GLUED.sub(r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- word-level swap
def _match_case(src, tgt):
    if src.isupper() and len(src) > 1:
        return tgt.upper()
    if src[:1].isupper():
        return tgt[:1].upper() + tgt[1:]
    return tgt


def _key(text):
    """Lower-case lookup key; titles like "Mr." keep their dot separately."""
    has_dot = len(text) > 1 and text.endswith(".")
    return (text[:-1] if has_dot else text).lower(), has_dot


def _in_slash_form(tok):
    doc = tok.doc
    return any(0 <= j < len(doc) and doc[j].text == "/" for j in (tok.i - 1, tok.i + 1))


def _is_title(tok, masked):
    """"Mr"/"Ms."/"Mrs"/"Miss" in title case, followed by a name (or the masked subject)."""
    text = tok.text.rstrip(".")
    if not (text[:1].isupper() and not text.isupper()):
        return False  # "MS" = Master of Science / Mississippi
    if tok.i + 1 >= len(tok.doc):
        return False
    nxt = tok.doc[tok.i + 1]
    return nxt.i in masked or nxt.text[:1].isupper()


def _swap_word(tok, masked):
    """Return the swapped surface form of `tok`, or None if it should not change."""
    if _in_slash_form(tok) or tok.ent_type_ in PROTECTED_ENTS:
        return None  # "s/he" is already neutral; "Women's Hospital" is a name
    key, has_dot = _key(tok.text)
    if key in TITLES:
        if not _is_title(tok, masked):
            return None
        new = "mr" if key in ("ms", "mrs", "miss") else "ms"
    elif key == "his":
        nxt = tok.nbor(1) if tok.i + 1 < len(tok.doc) else None
        new = "hers" if nxt is None or nxt.pos_ in _HIS_INDEPENDENT_NEXT else "her"
    elif key == "her":
        new = "his" if tok.tag_ == "PRP$" else "him"
    else:
        new = SWAP.get(key)
    if new is None:
        return None
    new = _match_case(tok.text, new)
    return new + "." if has_dot else new


def _is_leftover(tok, masked):
    """A gendered word that should have been swapped but was not."""
    if _in_slash_form(tok) or tok.ent_type_ in PROTECTED_ENTS:
        return False
    key, _ = _key(tok.text)
    if key in TITLES:
        return _is_title(tok, masked)
    if key in GENDERED_WORDS:
        return True
    # catch forms the tokenizer kept together, e.g. "him/herself"
    return any(part in GENDERED_WORDS for part in _SPLIT_RE.split(key) if part and part != key)


# ---------------------------------------------------------------- subject name
def _is_relative(span):
    """True if a kinship/role word sits within 3 tokens before the span."""
    doc = span.doc
    return any(t.lower_ in RELATIONAL for t in doc[max(0, span.start - 3):span.start])


def _subject_spans(doc):
    """PERSON entities that belong to the bio's subject (heuristic: the most frequent
    PERSON string that is not introduced as a relative, plus every PERSON span sharing a
    token with it, e.g. "Jane" in "Jane Doe")."""
    persons = [e for e in doc.ents if e.label_ == "PERSON"]
    candidates = [e for e in persons if not _is_relative(e)]
    if not candidates:
        return [], len(persons)
    top = Counter(e.text for e in candidates).most_common(1)[0][0]
    top_tokens = set(top.split())
    subj = [e for e in candidates if top_tokens & {t.text for t in e}]
    return subj, len(persons) - len(subj)


def _boilerplate_names(doc):
    """Directory boilerplate always names the subject, but NER often tags the name as an ORG:
      * "Call <name> on phone number ..."
      * "..., <name> affiliates with ..."  (e.g. "especially in NURSE PRACTITIONER, Patricia M Haggerty affiliates with")
    Return the token indices of <name>."""
    hits = set()
    for tok in doc:
        if tok.lower_ == "call":
            for j in range(tok.i + 2, min(tok.i + 8, len(doc) - 1)):
                if doc[j].lower_ == "on" and doc[j + 1].lower_ == "phone":
                    hits.update(range(tok.i + 1, j))
                    break
        elif tok.lower_ == "affiliates" and tok.i + 1 < len(doc) and doc[tok.i + 1].lower_ == "with":
            name, j = [], tok.i - 1
            while j >= 0 and (doc[j].text.istitle() or (len(doc[j].text) == 1 and doc[j].text.isupper())
                              or doc[j].text == "."):
                name.append(j)
                j -= 1
            if j >= 0 and doc[j].text == "," and 1 <= len(name) <= 5:
                hits.update(name)
    return hits


def _masked_indices(doc, subj):
    """Token indices to replace with {NAME}: the subject's PERSON spans, every other
    occurrence of the subject's name words that NER missed ("Kacy enjoys..."), outside
    protected entities such as "Smith Dental Clinic", and the name in the directory boilerplate
    "Call X on phone number" / ", X affiliates with"."""
    masked = {i for e in subj for i in range(e.start, e.end)} | _boilerplate_names(doc)
    name_words = {t.text for e in subj for t in e if t.is_alpha and len(t.text) > 1 and t.text[0].isupper()}
    for tok in doc:
        if tok.text in name_words and tok.ent_type_ not in PROTECTED_ENTS:
            masked.add(tok.i)
    return masked


_TITLE_WORDS = {"Dr", "Prof", "Mr", "Ms", "Mrs", "Miss", "St", "Sir"}


def _residual_names(doc, masked, first_names, stop, generic=frozenset()):
    """Unmasked tokens that are probably the subject's real name:
      1. a proper noun from the first-name list ("Jenifer has been working ..."), and
      2. any proper noun repeated 3+ times outside named organisations/places, which catches
         names missing from the list ("Gird extends herself ... Gird has earned ...").
    Words in `generic` (common lower-case words in the corpus, e.g. "surgery", "nurse") are never treated as repeated names."""
    def candidate(tok):
        return (tok.i not in masked and tok.pos_ == "PROPN" and tok.text.istitle()
                and tok.text.rstrip(".") not in _TITLE_WORDS and tok.text not in stop
                and tok.ent_type_ not in PROTECTED_ENTS)

    cands = [t for t in doc if candidate(t)]
    repeated = {w for w, n in Counter(t.text for t in cands).items()
                if n >= 3 and len(w) > 2 and w.lower() not in generic}
    hits = []
    for tok in cands:
        listed = (tok.text in first_names
                  and not (tok.i > 0 and doc[tok.i - 1].lower_ in ("of", "st", "st.", "saint", "san", "santa")))
        if listed or tok.text in repeated:
            hits.append(tok.text)
    return hits


# ---------------------------------------------------------------- main entry point
def swap_doc(doc, source_gender, mask_names=True, first_names=frozenset(), name_stop=frozenset(),
             generic_words=frozenset()):
    """Build the original and gender-swapped templates of one bio.

    `source_gender` is the gender the bio was written for and is only validated here.
    `first_names` (optional) enables the residual-name audit. `generic_words` (optional) are
    lower-case words that the repeated-name rule ignores. Returns a dict with
    template_orig, template_swap, n_swaps, n_subject_mentions, n_person_entities,
    n_other_persons, leftovers (gendered words not flipped) and residual_names.
    """
    if source_gender not in ("male", "female"):
        raise ValueError(f"source_gender must be 'male' or 'female', got {source_gender!r}")
    subj, n_other = _subject_spans(doc) if mask_names else ([], 0)
    masked = _masked_indices(doc, subj) if mask_names else set()

    orig_parts, swap_parts, leftovers = [], [], []
    n_swaps = 0
    for tok in doc:
        if tok.i in masked:
            # one {NAME} per run of adjacent masked tokens ("Jane Doe" -> "{NAME}")
            if tok.i + 1 not in masked:
                orig_parts.append("{NAME}" + tok.whitespace_)
                swap_parts.append("{NAME}" + tok.whitespace_)
            continue
        orig_parts.append(tok.text_with_ws)
        new = _swap_word(tok, masked)
        if new is None:
            swap_parts.append(tok.text_with_ws)
            if _is_leftover(tok, masked):
                leftovers.append(tok.text)
        else:
            n_swaps += 1
            swap_parts.append(new + tok.whitespace_)

    return {
        "template_orig": "".join(orig_parts),
        "template_swap": "".join(swap_parts),
        "n_swaps": n_swaps,
        "n_subject_mentions": len(subj),
        "n_person_entities": sum(e.label_ == "PERSON" for e in doc.ents),
        "n_other_persons": n_other,
        "leftovers": leftovers,
        "residual_names": _residual_names(doc, masked, first_names, name_stop, generic_words) if mask_names else [],
    }
