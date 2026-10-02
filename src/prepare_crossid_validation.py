"""Prepare silver-only attribution validation without gold authors or documents."""
import argparse
from collections import defaultdict
import csv
import hashlib
import heapq
import ijson
import json
from pathlib import Path
import random

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENRES = ('Article', 'Tweet')


def sha256(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def documents(path):
    with path.open('rb') as stream:
        yield from ijson.items(stream, 'item')


def prepare(silver_path, gold_path, output, authors=300, references_per_genre=30,
            targets_per_genre=15, seed=20260930):
    if authors < 2 or references_per_genre < 2 or references_per_genre % 2 or targets_per_genre < 1:
        raise ValueError('Need >=2 authors, an even reference count >=2, and >=1 target per genre.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Validation output must be a fresh directory.')
    excluded_authors, excluded_ids = set(), set()
    for doc in documents(gold_path):
        excluded_authors.add(str(doc['author']))
        excluded_ids.add(str(doc['id']))
    counts = defaultdict(lambda: defaultdict(int))
    seen = set()
    for doc in documents(silver_path):
        author, doc_id = str(doc['author']), str(doc['id'])
        if author in excluded_authors or doc_id in excluded_ids or doc['genre'] not in GENRES:
            continue
        if doc_id in seen:
            raise ValueError(f'Duplicate silver document ID: {doc_id}')
        seen.add(doc_id)
        if isinstance(doc.get('text'), str) and len(doc['text'].strip()) > 1:
            counts[author][doc['genre']] += 1
    needed = references_per_genre + targets_per_genre
    eligible = sorted(author for author, genres in counts.items()
                      if all(genres[genre] >= needed for genre in GENRES))
    if len(eligible) < authors:
        raise ValueError(f'Only {len(eligible)} eligible silver authors; requested {authors}.')
    del seen, counts
    rng = random.Random(seed)
    selected = sorted(rng.sample(eligible, authors))
    chosen = set(selected)
    pool = defaultdict(lambda: defaultdict(list))
    # A second streaming pass retains only the required documents per selected
    # author/genre. Seeded ID hashes make selection independent of input order.
    for doc in documents(silver_path):
        author, doc_id = str(doc['author']), str(doc['id'])
        if (author not in chosen or doc_id in excluded_ids or doc['genre'] not in GENRES
                or not isinstance(doc.get('text'), str) or len(doc['text'].strip()) <= 1):
            continue
        genre = doc['genre']
        priority = int(hashlib.sha256(f'{seed}\0{author}\0{genre}\0{doc_id}'.encode()).hexdigest(), 16)
        item = (-priority, doc_id, dict(id=doc_id, author=author, genre=genre, text=doc['text']))
        heap = pool[author][genre]
        if len(heap) < needed:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    refs, targets, both = [], [], []
    for author in selected:
        for genre in GENRES:
            docs = [item[2] for item in sorted(pool[author][genre], reverse=True)]
            reference = docs[:references_per_genre]
            refs.extend(reference)
            both.extend(reference[:references_per_genre // 2])
            targets.extend(docs[references_per_genre:needed])
    (output / 'query').mkdir(parents=True)
    (output / 'validation').mkdir()
    reference = pd.DataFrame(refs)
    for genre in GENRES:
        reference.loc[reference.genre == genre].to_csv(output / f'query/CrossNews_{genre}.csv',
                                                       index=False, quoting=csv.QUOTE_ALL)
    pd.DataFrame(both).to_csv(output / 'query/CrossNews_Both.csv', index=False, quoting=csv.QUOTE_ALL)
    pd.DataFrame(targets).to_csv(output / 'validation/CrossNews.csv', index=False, quoting=csv.QUOTE_ALL)
    protocol = {
        'role': 'silver_validation', 'seed': seed, 'authors': authors,
        'eligible_authors': len(eligible), 'author_labels': selected,
        'references_per_author_per_condition': references_per_genre,
        'targets_per_author': 2 * targets_per_genre,
        'excluded_gold_authors': sorted(excluded_authors),
        'sources': {str(path.resolve()): sha256(path) for path in [silver_path, gold_path]},
        'files': {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob('*.csv'))},
        'warning': 'Silver validation has different candidate authors; its raw accuracy is not a gold test result.',
    }
    (output / 'validation_protocol.json').write_text(json.dumps(protocol, indent=2))
    print(json.dumps({k: protocol[k] for k in ['authors', 'eligible_authors', 'seed']}, indent=2))
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--silver', type=Path, default=ROOT / 'raw_data/crossnews_silver.json')
    parser.add_argument('--gold', type=Path, default=ROOT / 'raw_data/crossnews_gold.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'crossid_validation_data')
    parser.add_argument('--authors', type=int, default=300)
    parser.add_argument('--references-per-genre', type=int, default=30)
    parser.add_argument('--targets-per-genre', type=int, default=15)
    parser.add_argument('--seed', type=int, default=20260930)
    args = parser.parse_args()
    try:
        prepare(args.silver, args.gold, args.output, args.authors,
                args.references_per_genre, args.targets_per_genre, args.seed)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
