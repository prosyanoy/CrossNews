"""Trained pipeline regression tests; synthetic metrics are not research results."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crossid_common import cohort, disjoint, identity, split_check, stylometry, STYLE_NAMES
from prepare_crossid_training import prepare

try:
    import torch
except ImportError:
    torch = None


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        docs = [dict(id=f'{a}-{g}-{i}', author=str(a), genre=g,
                     text=f'Author {a} genre {g} distinct document {i}. Words, style!')
                for a in range(8) for g in ('Article', 'Tweet') for i in range(4)]
        self.silver, self.gold = self.root / 'silver.json', self.root / 'gold.json'
        self.silver.write_text(json.dumps(docs))
        self.gold.write_text(json.dumps([dict(id='gold-id', author='gold-author', text='Gold text.')]))
        self.protocol = prepare(self.silver, self.gold, self.data, heldout_authors=2,
                                references=2, targets=2, seed=17)

    def test_authors_ids_and_texts_disjoint_across_roles(self):
        identities = []
        for stage in ('train', 'dev', 'calibration', 'test'):
            refs, targets, _ = cohort(self.data, stage)
            current = identity(refs, targets)
            for previous in identities:
                disjoint(previous, current)
            identities.append(current)
        with self.assertRaisesRegex(ValueError, 'authors'):
            disjoint(identities[0], identities[0])

    def test_split_tampering_rejected(self):
        path = self.data / 'test/test/CrossNews.csv'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'modified'):
            cohort(self.data, 'test')

    def test_gold_and_duplicate_text_exclusion_before_eligibility(self):
        docs = json.loads(self.silver.read_text())
        docs += [dict(id=f'gold-{g}-{i}', author='gold-author', genre=g, text=f'Unique excluded {g} {i}')
                 for g in ('Article', 'Tweet') for i in range(4)]
        docs += [dict(id='gold-dup-text', author='extra', genre='Tweet', text='  GOLD TEXT.  '),
                 dict(id='silver-dup-text', author='extra', genre='Article', text=docs[0]['text'].upper())]
        self.silver.write_text(json.dumps(docs))
        protocol = prepare(self.silver, self.gold, self.root / 'data2', heldout_authors=2,
                           references=2, targets=2, seed=17)
        self.assertEqual(protocol['excluded']['gold_identity'], 8)
        self.assertEqual(protocol['excluded']['repeated_text'], 2)
        self.assertNotIn('gold-author', sum(protocol['cohorts'].values(), []))

    def test_repeated_reference_target_text_requires_explicit_override(self):
        refs, targets, _ = cohort(self.data, 'test')
        targets = targets.copy()
        targets.loc[0, 'text'] = refs.iloc[0].text.upper()
        with self.assertRaisesRegex(ValueError, 'leak'):
            split_check(refs, targets)
        self.assertEqual(len(split_check(refs, targets, True)), 1)
        targets.loc[0, 'id'] = refs.iloc[0].id
        with self.assertRaisesRegex(ValueError, 'IDs'):
            split_check(refs, targets, True)

    def test_stylometry_handles_empty_unicode_and_punctuation(self):
        for value in ('', 'абв ABC!? @42\n', '   '):
            features = stylometry(value)
            self.assertEqual(len(features), len(STYLE_NAMES))
            self.assertTrue(np.isfinite(features).all())


@unittest.skipUnless(torch is not None, 'Install PyTorch for training tests')
class TrainingTests(ProtocolTests):
    def embeddings(self, stage):
        refs, targets, _ = cohort(self.data, stage)
        authors = sorted(refs.author.unique())
        paths = []
        rng = np.random.default_rng(7)
        for kind, frame in [('train', refs), ('target', targets)]:
            values = {}
            for row in frame.itertuples():
                vector = np.zeros(4)
                vector[authors.index(row.author)] = 1.
                vector[2] = .1 if row.genre == 'Article' else -.1
                vector += rng.normal(0, .01, 4)
                values[row.id] = vector.tolist()
            path = self.root / f'{stage}-{kind}.json'
            path.write_text(json.dumps(values))
            paths.append(path)
        return paths

    def args(self, command, output, stage='train', extra=()):
        from crossid_trained import parser
        refs, targets = self.embeddings(stage)
        options = [command, '--data-dir', str(self.data), '--train-embeddings', str(refs),
                   '--target-embeddings', str(targets), '--output', str(output), *extra]
        if command.startswith('train'):
            dr, dt = self.embeddings('dev')
            options += ['--dev-train-embeddings', str(dr), '--dev-target-embeddings', str(dt)]
        return parser().parse_args(options)

    def test_gradient_reversal_and_cross_genre_contrastive(self):
        from crossid_phase2 import Reverse, contrastive
        x = torch.tensor([2.], requires_grad=True)
        Reverse.apply(x, .3).sum().backward()
        self.assertAlmostEqual(float(x.grad), -.3, places=6)
        z = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1).requires_grad_()
        labels, genres = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])
        loss = contrastive(z, labels, genres, torch.tensor([0, 0, 0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        with self.assertRaisesRegex(ValueError, 'positive'):
            contrastive(z, labels, torch.zeros(4, dtype=torch.long), labels)

    def test_shortlist_only_reordering_and_misses_remain_outside(self):
        from crossid_phase3 import reorder
        self.assertEqual(reorder([2, 0, 1, 3], [2, 0], [.1, .9]), [0, 2, 1, 3])
        with self.assertRaisesRegex(ValueError, 'shortlist'):
            reorder([2, 0, 1, 3], [2, 3], [.1, .9])

    def test_reference_baseline_matches_phase1_and_selma_formula(self):
        from crossid_trained import baseline_scores
        from scipy.spatial.distance import cosine
        refs, _, _ = cohort(self.data, 'test')
        rng = np.random.default_rng(4)
        rx, tx = rng.normal(size=(len(refs), 4)), rng.normal(size=(3, 4))
        for local in (False, True):
            scores, authors = baseline_scores(refs, rx, tx, local)
            for i, target in enumerate(tx):
                for j, author in enumerate(authors):
                    matrix = rx[refs.author == author]
                    if local:
                        similarities = [1 - cosine(r, target) for r in matrix]
                        expected = np.mean(sorted(similarities)[-4:])
                    else:
                        expected = -round(cosine(matrix.mean(0), target), 4)
                    self.assertAlmostEqual(float(scores[i, j]), expected, places=6)

    def test_huggingface_local_save_load_and_gradient_path(self):
        try:
            from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
        except ImportError:
            self.skipTest('Install transformers for the local Hugging Face integration test')
        from crossid_phase3 import PairEncoder
        torch.set_num_threads(1)
        path = self.root / 'local-bert'
        path.mkdir()
        (path / 'vocab.txt').write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
                                                 'author', 'style', 'text', 'words', '.', '!']))
        tokenizer = BertTokenizerFast(vocab_file=str(path / 'vocab.txt'), do_lower_case=True)
        tokenizer.save_pretrained(path)
        BertForSequenceClassification(BertConfig(vocab_size=11, hidden_size=16, num_hidden_layers=1,
                                                num_attention_heads=2, intermediate_size=32,
                                                num_labels=1)).save_pretrained(path)
        encoder = PairEncoder(str(path), max_length=24, load=True)
        pairs = [('author text!', 'style words.'), ('style text.', 'author words!')]
        encoder.model.train()
        encoder.forward(pairs).sum().backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.model.parameters()))
        before = encoder.predict(pairs)
        saved = self.root / 'hf-saved'
        encoder.save(saved)
        restored = PairEncoder(str(saved), max_length=24, load=True)
        np.testing.assert_allclose(restored.predict(pairs), before, atol=1e-7)

    def test_end_to_end_training_calibration_and_frozen_evaluation(self):
        from crossid_phase2 import Adapter, load_adapter, train as train_adapter
        from crossid_phase3 import calibrate, directory_digest, load_fusion, train as train_reranker
        from crossid_trained import evaluate
        torch.set_num_threads(1)
        adapter, reranker, fusion = self.root / 'adapter', self.root / 'reranker', self.root / 'fusion'
        a = self.args('train-adapter', adapter, extra=['--epochs', '1', '--steps', '2',
                      '--authors-per-batch', '2', '--output-dim', '8', '--hidden-dim', '16',
                      '--prototypes', '2', '--topics', '2', '--max-bundle', '2'])
        meta = train_adapter(a)
        torch.manual_seed(a.seed)
        initial = Adapter(4, 8, 16, 2)
        trained, _ = load_adapter(adapter)
        self.assertTrue(any(not torch.equal(v, trained.state_dict()[k]) for k, v in initial.state_dict().items()))
        self.assertTrue(np.isfinite(meta['history'][0]['loss']))
        r = self.args('train-reranker', reranker, extra=['--adapter', str(adapter), '--encoder', 'tiny',
                      '--epochs', '1', '--max-length', '48', '--batch-size', '4', '--shortlist', '2'])
        train_reranker(r)
        c = self.args('calibrate', fusion, stage='calibration', extra=['--adapter', str(adapter),
                      '--reranker', str(reranker)])
        calibrate(c)
        before = directory_digest(adapter), directory_digest(reranker), directory_digest(fusion)
        e = self.args('evaluate', self.root / 'results', stage='test', extra=['--adapter', str(adapter),
                      '--reranker', str(reranker), '--fusion', str(fusion / 'fusion.json')])
        summary = evaluate(e)
        self.assertEqual(set(summary), {'selma', 'reference_only', 'adapter_reference_only',
                                      'adapter_prototypes', 'cross_encoder', 'fused'})
        self.assertEqual(summary['fused']['Overall']['n'], 8)
        self.assertEqual(json.loads((e.output / 'manifest.json').read_text())['status'], 'completed')
        self.assertEqual(before, (directory_digest(adapter), directory_digest(reranker), directory_digest(fusion)))
        with self.assertRaisesRegex(ValueError, 'Reference condition'):
            load_fusion(fusion / 'fusion.json', adapter, reranker, 'Article')
        # A modified checkpoint cannot be silently evaluated or calibrated.
        with (adapter / 'adapter.pt').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'weights changed'):
            load_adapter(adapter)


if __name__ == '__main__':
    unittest.main()
