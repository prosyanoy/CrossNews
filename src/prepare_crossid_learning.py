"""Prepare disjoint silver author splits for CROSS-ID Phases 2 and 3."""
import argparse
from collections import defaultdict
import csv
import hashlib
import heapq
import json
from pathlib import Path
import random
import sqlite3
import tempfile

import pandas as pd

from prepare_crossid_validation import documents
from crossid_learning.common import (GENRES, ROOT, digest, fresh_directory,
                                     text_hash, write_json)


def prepare(silver, gold, output, dev_authors=60, calibration_authors=60,
            test_authors=60, references=30, targets=15, train_per_genre=100,
            seed=20261001):
    silver, gold, output = Path(silver), Path(gold), Path(output)
    if min(dev_authors, calibration_authors, test_authors) < 2:
        raise ValueError('Each held-out split needs at least two authors.')
    if references < 2 or references % 2 or targets < 1 or train_per_genre < references + targets:
        raise ValueError('Need even references >=2, targets >=1, and enough training documents.')
    if Path(output).exists() and any(Path(output).iterdir()):
        raise ValueError('Preparation requires a fresh output directory.')
    output.parent.mkdir(parents=True, exist_ok=True)
    excluded_authors, excluded_ids, excluded_texts = set(), set(), set()
    for doc in documents(gold):
        excluded_authors.add(str(doc['author'])); excluded_ids.add(str(doc['id']))
        if isinstance(doc.get('text'), str):
            excluded_texts.add(text_hash(doc['text']))
    # A disk index selects one canonical ID per normalized text. This prevents
    # syndicated/duplicated text crossing authors or roles without keeping the
    # silver corpus in RAM. Selection remains independent of JSON record order.
    with tempfile.TemporaryDirectory(prefix='crossid-index-', dir=Path(output).parent) as temporary:
        db = sqlite3.connect(str(Path(temporary) / 'documents.sqlite'))
        db.execute('CREATE TABLE docs (id TEXT PRIMARY KEY, hash TEXT, author TEXT, genre TEXT)')
        batch = []
        for doc in documents(silver):
            author, doc_id, genre = str(doc['author']), str(doc['id']), doc['genre']
            text = doc.get('text')
            if (author in excluded_authors or doc_id in excluded_ids or genre not in GENRES
                    or not isinstance(text, str) or len(text.strip()) < 2):
                continue
            h = text_hash(text)
            if h in excluded_texts:
                continue
            batch.append((doc_id, h, author, genre))
            if len(batch) == 10000:
                db.executemany('INSERT INTO docs VALUES (?,?,?,?)', batch); batch = []
        try:
            db.executemany('INSERT INTO docs VALUES (?,?,?,?)', batch)
            db.execute('CREATE TABLE canonical AS SELECT min(id) AS id FROM docs GROUP BY hash')
            db.execute('CREATE UNIQUE INDEX canonical_id ON canonical(id)')
            counts = defaultdict(dict)
            for author, genre, count in db.execute(
                    'SELECT author,genre,count(*) FROM docs JOIN canonical USING(id) GROUP BY author,genre'):
                counts[author][genre] = count
            needed = references + targets
            eligible = sorted(a for a, c in counts.items() if all(c.get(g, 0) >= needed for g in GENRES))
            if len(eligible) < dev_authors + calibration_authors + test_authors + 2:
                raise ValueError(f'Only {len(eligible)} eligible non-gold, deduplicated authors; '
                                 'reduce held-out counts while retaining >=2 training authors.')
            rng = random.Random(seed); rng.shuffle(eligible)
            roles, offset = {}, 0
            for role, count in [('dev', dev_authors), ('calibration', calibration_authors), ('test', test_authors)]:
                roles[role] = sorted(eligible[offset:offset+count]); offset += count
            roles['train'] = sorted(eligible[offset:])
            role_for_author = {a: role for role, authors in roles.items() for a in authors}
            pool = defaultdict(list)
            for doc in documents(silver):
                author, doc_id, genre = str(doc['author']), str(doc['id']), doc['genre']
                if author not in role_for_author or genre not in GENRES:
                    continue
                if not db.execute('SELECT 1 FROM canonical WHERE id=?', (doc_id,)).fetchone():
                    continue
                limit = train_per_genre if role_for_author[author] == 'train' else needed
                priority = int(hashlib.sha256(f'{seed}\0{doc_id}'.encode()).hexdigest(), 16)
                item = (-priority, doc_id, dict(id=doc_id, author=author, genre=genre, text=doc['text']))
                heap = pool[(author, genre)]
                if len(heap) < limit:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
        finally:
            db.close()
    output = fresh_directory(output)
    refs_for_embeddings, targets_for_embeddings = [], []
    for role, authors in roles.items():
        refs, both, unknown, all_docs = [], [], [], []
        for author in authors:
            for genre in GENRES:
                docs = [item[2] for item in sorted(pool[(author, genre)], reverse=True)]
                refs.extend(docs[:references]); both.extend(docs[:references//2])
                unknown.extend(docs[references:needed]); all_docs.extend(docs)
        folder = output / role; (folder / 'query').mkdir(parents=True)
        for genre in GENRES:
            pd.DataFrame([d for d in refs if d['genre'] == genre]).to_csv(
                folder / 'query' / f'CrossNews_{genre}.csv', index=False, quoting=csv.QUOTE_ALL)
        pd.DataFrame(both).to_csv(folder / 'query/CrossNews_Both.csv', index=False, quoting=csv.QUOTE_ALL)
        pd.DataFrame(unknown).to_csv(folder / 'targets.csv', index=False, quoting=csv.QUOTE_ALL)
        if role == 'train':
            pd.DataFrame(all_docs).to_csv(folder / 'documents.csv', index=False, quoting=csv.QUOTE_ALL)
            refs_for_embeddings.extend(all_docs); targets_for_embeddings.extend(all_docs)
        else:
            refs_for_embeddings.extend(refs); targets_for_embeddings.extend(unknown)
    for name, rows in [('embedding_references.csv', refs_for_embeddings),
                       ('embedding_targets.csv', targets_for_embeddings)]:
        pd.DataFrame(rows).to_csv(output / name, index=False, quoting=csv.QUOTE_ALL)
    protocol = {'role': 'crossid_phase23_silver', 'status': 'completed', 'seed': seed,
                'author_splits': roles, 'eligible_authors': len(eligible),
                'excluded_gold_authors': sorted(excluded_authors),
                'references_per_condition': references, 'targets_per_genre': targets,
                'train_per_genre_cap': train_per_genre,
                'deduplication': 'canonical minimum ID per casefolded whitespace-normalized text; exclude gold texts',
                'sources': {str(p.resolve()): digest(p) for p in [silver, gold]},
                'files': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*.csv'))},
                'note': 'A new protocol; any earlier Phase 1 silver validation author may now be in training. '
                        'Use only this protocol for independent Phase 2/3 testing. Gold results remain exploratory.'}
    write_json(output / 'learning_protocol.json', protocol)
    print(json.dumps({role: len(authors) for role, authors in roles.items()}, indent=2))
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--silver', type=Path, default=ROOT / 'raw_data/crossnews_silver.json')
    p.add_argument('--gold', type=Path, default=ROOT / 'raw_data/crossnews_gold.json')
    p.add_argument('--output', type=Path, default=ROOT / 'crossid_learning_data')
    p.add_argument('--dev-authors', type=int, default=60)
    p.add_argument('--calibration-authors', type=int, default=60)
    p.add_argument('--test-authors', type=int, default=60)
    p.add_argument('--references', type=int, default=30)
    p.add_argument('--targets', type=int, default=15)
    p.add_argument('--train-per-genre', type=int, default=100)
    p.add_argument('--seed', type=int, default=20261001)
    args = p.parse_args()
    try:
        prepare(args.silver, args.gold, args.output, args.dev_authors, args.calibration_authors,
                args.test_authors, args.references, args.targets, args.train_per_genre, args.seed)
    except (ValueError, FileNotFoundError, sqlite3.IntegrityError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
