"""Deterministic silver train/dev/calibration/test authors, with gold exclusion."""
import argparse
from collections import defaultdict
import csv
import hashlib
import heapq
from pathlib import Path
import random

import pandas as pd

from crossid_common import STAGES, fresh, text_hash, write_json
from prepare_crossid_validation import documents, sha256


def prepare(silver, gold, output, heldout_authors=60, references=30, targets=15, seed=20261001):
    if heldout_authors < 2 or references < 2 or references % 2 or targets < 1:
        raise ValueError('Need >=2 held-out authors, even references >=2 and targets >=1.')
    gold_authors, gold_ids, seen_texts = set(), set(), set()
    for doc in documents(Path(gold)):
        gold_authors.add(str(doc['author']))
        gold_ids.add(str(doc['id']))
        if isinstance(doc.get('text'), str):
            seen_texts.add(text_hash(doc['text']))
    pool = defaultdict(lambda: defaultdict(list))
    seen_ids = set()
    needed = references + targets
    # Globally deduplicate before counting eligibility; deterministic for this
    # hashed source file. Input ordering is recorded, never silently changed.
    excluded = defaultdict(int)
    for doc in documents(Path(silver)):
        author, doc_id = str(doc['author']), str(doc['id'])
        genre, text = doc.get('genre'), doc.get('text')
        if author in gold_authors or doc_id in gold_ids:
            excluded['gold_identity'] += 1
            continue
        if genre not in ('Article', 'Tweet') or not isinstance(text, str) or len(text.strip()) < 2:
            excluded['invalid'] += 1
            continue
        if doc_id in seen_ids:
            raise ValueError(f'Duplicate silver ID: {doc_id}')
        seen_ids.add(doc_id)
        fingerprint = text_hash(text)
        if fingerprint in seen_texts:
            excluded['repeated_text'] += 1
            continue
        seen_texts.add(fingerprint)
        priority = int(hashlib.sha256(f'{seed}\0{doc_id}'.encode()).hexdigest(), 16)
        item = (-priority, doc_id, dict(id=doc_id, author=author, genre=genre, text=text))
        heap = pool[author][genre]
        if len(heap) < needed:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    eligible = sorted(a for a, genres in pool.items()
                      if all(len(genres[g]) == needed for g in ('Article', 'Tweet')))
    if len(eligible) < 3 * heldout_authors + 2:
        raise ValueError(f'{len(eligible)} eligible authors; need at least {3 * heldout_authors + 2}.')
    random.Random(seed).shuffle(eligible)
    cohorts = {s: sorted(eligible[i * heldout_authors:(i + 1) * heldout_authors])
               for i, s in enumerate(STAGES[1:])}
    cohorts['train'] = sorted(eligible[3 * heldout_authors:])
    output = fresh(output)
    for stage in STAGES:
        refs, queries = [], []
        for author in cohorts[stage]:
            for genre in ('Article', 'Tweet'):
                docs = [v[2] for v in sorted(pool[author][genre], reverse=True)]
                refs.extend(docs[:references])
                queries.extend(docs[references:])
        stage_path = output / stage
        (stage_path / 'query').mkdir(parents=True)
        (stage_path / 'test').mkdir()
        frame = pd.DataFrame(refs)
        for genre in ('Article', 'Tweet'):
            frame[frame.genre == genre].to_csv(stage_path / f'query/CrossNews_{genre}.csv',
                                               index=False, quoting=csv.QUOTE_ALL)
        both = frame.groupby(['author', 'genre'], sort=False).head(references // 2)
        both.to_csv(stage_path / 'query/CrossNews_Both.csv', index=False, quoting=csv.QUOTE_ALL)
        pd.DataFrame(queries).to_csv(stage_path / 'test/CrossNews.csv', index=False, quoting=csv.QUOTE_ALL)
    protocol = {'format': 'crossid_training_protocol_v1', 'seed': seed,
                'cohorts': cohorts, 'eligible_authors': len(eligible),
                'references_per_genre': references, 'targets_per_genre': targets,
                'excluded': dict(excluded), 'excluded_gold_authors': sorted(gold_authors),
                'deduplication': 'Global casefolded whitespace-normalized text SHA256; first occurrence retained.',
                'sources': {str(Path(p).resolve()): sha256(Path(p)) for p in (silver, gold)},
                'files': {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*.csv'))}}
    write_json(output / 'protocol.json', protocol)
    print({s: len(cohorts[s]) for s in STAGES})
    return protocol


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--silver', type=Path, default=Path('raw_data/crossnews_silver.json'))
    p.add_argument('--gold', type=Path, default=Path('raw_data/crossnews_gold.json'))
    p.add_argument('--output', type=Path, default=Path('crossid_training_data'))
    p.add_argument('--heldout-authors', type=int, default=60)
    p.add_argument('--references', type=int, default=30)
    p.add_argument('--targets', type=int, default=15)
    p.add_argument('--seed', type=int, default=20261001)
    a = p.parse_args()
    try:
        prepare(a.silver, a.gold, a.output, a.heldout_authors, a.references, a.targets, a.seed)
    except (ValueError, FileNotFoundError) as e:
        p.error(str(e))
