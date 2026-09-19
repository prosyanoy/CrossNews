"""Synthetic regression tests. These are not CrossNews benchmark results."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from attribution_models.crossid import CrossID
from benchmark_crossid import parser, preflight, run
from utils import evaluate_attribution_scores


class CrossIDTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        (self.data / 'query').mkdir(parents=True)
        (self.data / 'test').mkdir()
        references, targets, train, test = [], [], {}, {}
        for author, vector in [('alice', [1., 0.]), ('bob', [0., 1.])]:
            for genre in ['Article', 'Tweet']:
                for i in range(10):
                    doc = f'{author}_{genre}_{i}'
                    references.append(dict(id=doc, author=author, genre=genre))
                    train[doc] = vector
                doc = f'target_{author}_{genre}'
                targets.append(dict(id=doc, author=author, genre=genre))
                test[doc] = vector
        refs = pd.DataFrame(references)
        for genre in ['Article', 'Tweet', 'Both']:
            df = refs if genre == 'Both' else refs[refs.genre == genre]
            df.to_csv(self.data / f'query/CrossNews_{genre}.csv', index=False)
        pd.DataFrame(targets).to_csv(self.data / 'test/CrossNews.csv', index=False)
        self.train_path, self.test_path = self.root / 'train.json', self.root / 'test.json'
        self.train_path.write_text(json.dumps(train))
        self.test_path.write_text(json.dumps(test))
        self.params = dict(name='crossid', train_embedding_loc=str(self.train_path),
                           test_embedding_loc=str(self.test_path), num_prototypes=1)
        self.args = SimpleNamespace(query_file=str(self.data / 'query/CrossNews_Both.csv'),
                                    target_file=str(self.data / 'test/CrossNews.csv'),
                                    train=True, test=True, load=False, save_folder=str(self.root / 'models'))

    def model(self, **overrides):
        return CrossID(self.args, self.params | overrides)

    def test_scores_and_saved_configuration(self):
        model = self.model()
        result, authors = model.evaluate()
        self.assertEqual(authors, ['alice', 'bob'])
        self.assertTrue(all(r['label'] == r['prediction'] and r['rank'] == 1 for r in result))
        model.train()
        self.args.load = True
        self.args.load_folder = model.model_folder
        loaded = CrossID(self.args, {'invalid': 'ignored in load mode'})
        self.assertEqual(result, loaded.evaluate()[0])

    def test_cache_invalidated_after_sixteenth_document(self):
        model = self.model()
        before = model._build_author_profiles(model.query_df)
        changed = model.query_df.copy()
        # Same first 16 IDs, same size, same author set, changed later assignment.
        changed.loc[19, 'author'] = 1
        after = model._build_author_profiles(changed)
        self.assertIsNot(before, after)
        self.assertEqual(len(after[0]['references']), 19)

    def test_ties_have_one_winner(self):
        model = self.model()
        model.id_to_embedding = {k: np.array([1., 0.]) for k in model.id_to_embedding}
        results, _ = model.evaluate()
        metrics = evaluate_attribution_scores(results)
        accuracy = sum(r['label'] == r['prediction'] for r in results) / len(results)
        self.assertEqual(metrics['Accuracy'], accuracy)
        self.assertEqual(accuracy, .5)

    def test_invalid_weights_and_counts(self):
        for overrides in [dict(prototype_weight=-1), dict(centroid_weight=float('nan')),
                          dict(prototype_weight=0, centroid_weight=0, reference_weight=0),
                          dict(reference_top_k=0), dict(num_prototypes=0)]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.model(**overrides)

    def test_invalid_embeddings(self):
        for vector in [[0, 0], [float('nan'), 1], [1, 2, 3], [[1, 2]]]:
            with self.subTest(vector=vector):
                data = json.loads(self.test_path.read_text())
                data['target_alice_Article'] = vector
                self.test_path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    self.model()

    def test_overlap_rejected(self):
        model = self.model()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            model.evaluate_internal(model.query_df, model.query_df)

    def test_kmeans_and_weighted_score(self):
        model = self.model(num_prototypes=2, prototype_top_k=1, reference_top_k=1)
        matrix = np.array([[1, 0], [.99, .01], [0, 1], [.01, .99]], dtype=np.float32)
        first, second = model._make_profile(matrix), model._make_profile(matrix)
        np.testing.assert_allclose(first['prototypes'], second['prototypes'])
        target = np.array([1, 0], dtype=np.float32)
        expected = (.55 * max(first['prototypes'] @ target)
                    + .20 * (first['centroid'] @ target)
                    + .25 * max(first['references'] @ target))
        self.assertAlmostEqual(model._score_profile(target, first), expected, places=6)

    def benchmark_args(self):
        return parser().parse_args(['--data-dir', str(self.data), '--train-embeddings', str(self.train_path),
                                    '--test-embeddings', str(self.test_path), '--output', str(self.root / 'benchmark')])

    def test_benchmark_end_to_end(self):
        args = self.benchmark_args()
        # Avoid degenerate KMeans fixtures while still exercising both configurations.
        params = self.root / 'parameters.json'
        params.write_text(json.dumps({'default': self.params, 'prototype_heavy': self.params}))
        args.parameters = params
        run(args)
        results = json.loads((args.output / 'summary.json').read_text())
        self.assertEqual(len(results), 27)
        self.assertTrue(all(r['Accuracy'] == 1 for r in results))
        self.assertEqual(json.loads((args.output / 'manifest.json').read_text())['status'], 'completed')
        with self.assertRaisesRegex(ValueError, 'not empty'):
            run(args)

    def test_missing_embeddings_preflight(self):
        self.test_path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing benchmark inputs'):
            preflight(self.benchmark_args())

    def test_runner_multiple_parameter_sets(self):
        # Existing runner loads repository configs by relative path.
        (self.root / 'src/model_parameters').mkdir(parents=True)
        (self.root / 'src/model_parameters/crossid.json').write_text(json.dumps({
            'default': self.params, 'prototype_heavy': self.params}))
        completed = subprocess.run([sys.executable, str(ROOT / 'src/run_attribution.py'), '--model', 'crossid',
                                    '--train', '--test', '--query_file', self.args.query_file,
                                    '--target_file', self.args.target_file, '--parameter_sets',
                                    'default', 'prototype_heavy', '--save_folder', str(self.root / 'cli')],
                                   cwd=self.root, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(list((self.root / 'cli').rglob('test_results.json'))), 2)


if __name__ == '__main__':
    unittest.main()
