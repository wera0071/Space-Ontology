#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def load_concepts(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of concepts")
        return data

    concepts = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            concepts.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Bad JSON at line {lineno}: {e}") from e
    return concepts


def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("Ё", "Е").replace("ё", "е")
    s = s.replace("“", '"').replace("”", '"').replace("„", '"')
    s = s.replace("«", '"').replace("»", '"')
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_key(s: str) -> str:
    s = clean_text(s).lower()
    s = re.sub(r"[/\\|+_=~*^#@%$]+", " ", s)
    s = re.sub(r"[.,:;!?()\[\]{}\"'`]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def get_morph():
    try:
        import pymorphy3
    except ImportError:
        return None
    return pymorphy3.MorphAnalyzer()


def _is_acronym_or_proper(token_orig: str) -> bool:
    """True если токен не надо лемматизировать (аббревиатура/имя собств./латиница/цифры)."""
    if not token_orig:
        return True
    if re.search(r"\d", token_orig):
        return True
    if re.search(r"[A-Za-z]", token_orig):
        return True
    cyr = re.findall(r"[А-Яа-яЁё]", token_orig)
    if not cyr:
        return True
    # все кириллические буквы заглавные и токен >= 2 букв => аббревиатура
    if len(cyr) >= 2 and all(ch.isupper() for ch in cyr):
        return True
    return False


def _safe_lemma(token_lower: str, morph) -> str:
    """Лемматизация с защитой: отказываемся, если pymorphy сильно искажает токен."""
    parsed = morph.parse(token_lower)[0]
    lemma = parsed.normal_form
    # если лемма стала сильно короче (как БПЛА -> бпнуть/бпать) — оставляем как есть
    if len(token_lower) >= 4 and len(lemma) < len(token_lower) * 0.6:
        return token_lower
    return lemma


def lemmatize_phrase(s: str, morph=None) -> str:
    s = clean_text(s)
    if not s:
        return ""
    s_norm = re.sub(r"[/\\|+_=~*^#@%$]+", " ", s)
    s_norm = re.sub(r"[.,:;!?()\[\]{}\"\'`]+", " ", s_norm)
    s_norm = re.sub(r"\s+", " ", s_norm).strip()
    if not s_norm:
        return ""
    if morph is None:
        return s_norm.lower()

    out = []
    for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", s_norm):
        if _is_acronym_or_proper(token):
            out.append(token.lower())
            continue
        parts = [p for p in token.split("-") if p]
        if len(parts) > 1:
            sub = []
            for p in parts:
                if _is_acronym_or_proper(p):
                    sub.append(p.lower())
                else:
                    sub.append(_safe_lemma(p.lower(), morph))
            out.append("-".join(sub))
        else:
            out.append(_safe_lemma(token.lower(), morph))
    return " ".join(out).strip()


def term_keys(term: str, morph=None) -> Set[str]:
    keys = set()
    raw = normalize_key(term)
    if raw:
        keys.add(raw)
        no_hyphen = raw.replace("-", " ")
        keys.add(re.sub(r"\s+", " ", no_hyphen).strip())
    lemma = lemmatize_phrase(term, morph)
    if lemma:
        keys.add(lemma)
        keys.add(re.sub(r"\s+", " ", lemma.replace("-", " ")).strip())
    return {k for k in keys if k}


def concept_id(concept: dict) -> str:
    return str(concept.get("conceptid", "")).strip()


def concept_name(concept: dict) -> str:
    return clean_text(str(concept.get("conceptstr", "")))


def iter_terms(concept: dict, include_english: bool = False) -> Iterable[Tuple[str, str]]:
    name = concept_name(concept)
    if name:
        yield name, "conceptstr"

    for syn in concept.get("synonyms") or []:
        for field in ("textentrystr", "lementrystr"):
            val = clean_text(str(syn.get(field, "")))
            if val:
                yield val, f"synonyms.{field}"

    if include_english:
        eng = clean_text(str(concept.get("conceptengstr", "")))
        if eng:
            yield eng, "conceptengstr"
        for syn in concept.get("e_synonyms") or []:
            for field in ("textentrystr", "lementrystr"):
                val = clean_text(str(syn.get(field, "")))
                if val:
                    yield val, f"e_synonyms.{field}"


def build_term_index(
    concepts: Sequence[dict],
    include_english: bool = False,
    use_morph: bool = True,
    include_relats: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, dict], Dict[str, List[Tuple[str, str]]]]:
    morph = get_morph() if use_morph else None
    index: Dict[str, List[str]] = defaultdict(list)
    concept_by_id: Dict[str, dict] = {}
    debug_terms: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for concept in concepts:
        cid = concept_id(concept)
        if not cid:
            continue
        concept_by_id[cid] = concept
        for term, source in iter_terms(concept, include_english=include_english):
            for key in term_keys(term, morph):
                if cid not in index[key]:
                    index[key].append(cid)
                debug_terms[key].append((cid, source))

        if include_relats:
            for rel in concept.get("relats") or []:
                rel_cid = str(rel.get("conceptid", "")).strip()
                rel_name = clean_text(str(rel.get("conceptstr", "")))
                if not rel_cid or not rel_name:
                    continue
                concept_by_id.setdefault(
                    rel_cid,
                    {
                        "conceptid": rel_cid,
                        "conceptstr": rel_name,
                        "synonyms": [],
                        "e_synonyms": [],
                        "relats": [],
                    },
                )
                for key in term_keys(rel_name, morph):
                    if rel_cid not in index[key]:
                        index[key].append(rel_cid)
                    debug_terms[key].append((rel_cid, "relats.conceptstr"))

    return dict(index), concept_by_id, dict(debug_terms)


def direct_parents(concept: dict, relation: str = "ВЫШЕ") -> List[str]:
    out = []
    for rel in concept.get("relats") or []:
        if clean_text(str(rel.get("relationstr", ""))).upper() != relation.upper():
            continue
        parent_id = str(rel.get("conceptid", "")).strip()
        if parent_id and parent_id not in out:
            out.append(parent_id)
    return out


def build_parent_graph(concepts: Sequence[dict], relation: str = "ВЫШЕ") -> Dict[str, List[str]]:
    graph: Dict[str, List[str]] = {}
    for concept in concepts:
        cid = concept_id(concept)
        if cid:
            graph[cid] = direct_parents(concept, relation=relation)
    return graph


def ancestors(concept_id_value: str, graph: Dict[str, List[str]], max_depth: int) -> List[str]:
    if max_depth <= 0:
        return []

    seen = set()
    out = []
    q = deque([(str(concept_id_value), 0)])
    while q:
        cid, depth = q.popleft()
        if depth >= max_depth:
            continue
        for parent in graph.get(cid, []):
            if parent in seen:
                continue
            seen.add(parent)
            out.append(parent)
            q.append((parent, depth + 1))
    return out


def resolve_term(term: str, index: Dict[str, List[str]], use_morph: bool = True) -> Tuple[List[str], str]:
    morph = get_morph() if use_morph else None
    for key in term_keys(term, morph):
        ids = index.get(key, [])
        if ids:
            return ids, key
    return [], ""
