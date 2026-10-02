"""Genre ablations and validation/test separation regression tests."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import test_crossid as fixtures
from benchmark_crossid import parser, preflight, run
from prepare_crossid_validation import prepare
from select_crossid_validation import freeze


class AblationTests(unittest.TestCase):
    setUp = fixtures.CrossIDTests.setUp
    model = fixtures.CrossIDTests.model

    def genre_model(self, mode, **kwargs):
        train = json.loads(self.train_path.read_text())
        for doc_id in train:
            train[doc_id] = [1., 0.] if '_Article_' in doc_id else [0., 1.]
        self.train_path.write_text(json.dumps(train))
        return self.model(profile_mode=mode, num_prototypes=2, **kwargs)

    def test_prototype_allocation_isolates_centroid_and_references(self):
        pooled = self.genre_model('pooled')
        balanced = self.genre_model('genre_prototypes')
        p = pooled._build_author_profiles(pooled.query_df)[0]
        b = balanced._build_author_profiles(balanced.query_df)[0]
        self.assertEqual(len(b['prototypes']), 2)
        np.testing.assert_allclose(p['centroid'], b['centroid'])
        np.testing.assert_allclose(p['references'], b['references'])
        self.assertEqual({tuple(v) for v in b['prototypes']}, {(1., 0.), (0., 1.)})

    def test_balanced_scores_do_not_weight_larger_genre_bundles(self):
        model = self.genre_model('genre_balanced', prototype_weight=0, centroid_weight=1, reference_weight=0)
        query = model.query_df.drop(index=[15, 16, 17, 18, 19])
        profile = model._build_author_profiles(query)[0]
        self.assertAlmostEqual(model._score_profile(np.array([1., 0.]), profile), .5)

    def test_matched_scores_use_observed_genre_and_cross_genre_fallback(self):
        model = self.genre_model('genre_matched', prototype_weight=0, centroid_weight=1, reference_weight=0)
        profile = model._build_author_profiles(model.query_df)[0]
        self.assertAlmostEqual(model._score_profile(np.array([1., 0.]), profile, 'Article'), 1.)
        self.assertAlmostEqual(model._score_profile(np.array([1., 0.]), profile, 'Tweet'), 0.)
        article = model.query_df[model.query_df.genre == 'Article']
        profile = model._build_author_profiles(article)[0]
        self.assertAlmostEqual(model._score_profile(np.array([1., 0.]), profile, 'Tweet'), 1.)

    def test_genre_change_invalidates_cache(self):
        model = self.genre_model('genre_balanced')
        before = model._build_author_profiles(model.query_df)
        changed = model.query_df.copy()
        changed.loc[19, 'genre'] = 'Article'
        after = model._build_author_profiles(changed)
        self.assertIsNot(before, after)
        self.assertEqual(len(after[0]['genres']['Article']['references']), 11)

    def test_centroid_reference_ablations_skip_kmeans(self):
        for weights in [(0, 1, 0), (0, 0, 1)]:
            model = self.model(num_prototypes=4, prototype_weight=weights[0],
                               centroid_weight=weights[1], reference_weight=weights[2])
            with patch('attribution_models.crossid.KMeans', side_effect=AssertionError('unused KMeans')):
                self.assertTrue(all(r['label'] == r['prediction'] for r in model.evaluate()[0]))

    def prepare_validation(self, folder):
        docs = [dict(id=f's{a}_{genre}_{i}', author=f's{a}', genre=genre, text=f'Example document {i}')
                for a in range(4) for genre in ['Article', 'Tweet'] for i in range(4)]
        # This silver author and this reused ID must never enter validation.
        docs += [dict(id=f'gold_{genre}_{i}', author='gold', genre=genre, text='Gold author text')
                 for genre in ['Article', 'Tweet'] for i in range(4)]
        silver, gold = self.root / 'silver.json', self.root / 'gold.json'
        silver.write_text(json.dumps(docs))
        gold.write_text(json.dumps([dict(author='gold', id='s0_Article_0')]))
        return prepare(silver, gold, folder, authors=2, references_per_genre=2, targets_per_genre=1, seed=7)

    def test_validation_excludes_gold_and_is_independent_of_input_order(self):
        first, second = self.root / 'val1', self.root / 'val2'
        self.prepare_validation(first)
        silver = self.root / 'silver.json'
        silver.write_text(json.dumps(list(reversed(json.loads(silver.read_text())))))
        prepare(silver, self.root / 'gold.json', second, authors=2,
                references_per_genre=2, targets_per_genre=1, seed=7)
        for path in first.rglob('*.csv'):
            self.assertEqual(path.read_bytes(), (second / path.relative_to(first)).read_bytes())
            frame = pd.read_csv(path)
            self.assertNotIn('gold', set(frame.author))
            self.assertNotIn('s0_Article_0', set(frame.id))
        refs = pd.concat([pd.read_csv(first / f'query/CrossNews_{g}.csv') for g in ['Article', 'Tweet']])
        targets = pd.read_csv(first / 'validation/CrossNews.csv')
        self.assertFalse(set(refs.id) & set(targets.id))

    def validation_run(self):
        data = self.root / 'validation_data'
        protocol = self.prepare_validation(data)
        authors = protocol['author_labels']
        paths = []
        for split, frames in [('train', [data / f'query/CrossNews_{g}.csv' for g in ['Article', 'Tweet']]),
                              ('test', [data / 'validation/CrossNews.csv'])]:
            embeddings = {}
            for frame_path in frames:
                for row in pd.read_csv(frame_path).itertuples():
                    embeddings[row.id] = [1., 0.] if row.author == authors[0] else [0., 1.]
            path = self.root / f'val_{split}.json'
            path.write_text(json.dumps(embeddings))
            paths.append(path)
        configs = self.root / 'validation_configs.json'
        configs.write_text(json.dumps({'pooled': self.params, 'balanced': self.params | {
            'num_prototypes': 2, 'profile_mode': 'genre_balanced'}}))
        args = parser().parse_args(['--data-dir', str(data), '--train-embeddings', str(paths[0]),
                                    '--test-embeddings', str(paths[1]), '--parameters', str(configs),
                                    '--parameter-sets', 'all', '--reference-sets', 'Both',
                                    '--split', 'validation', '--output', str(self.root / 'validation_results')])
        run(args)
        selection_path = self.root / 'selection.json'
        selection = freeze(args.output, selection_path)
        return args, selection_path, selection

    def test_validation_selection_then_disjoint_test(self):
        args, selection_path, selection = self.validation_run()
        self.assertEqual(selection['selected']['Both']['model'], 'balanced')  # deterministic tie rule
        test_args = parser().parse_args(['--data-dir', str(self.data), '--train-embeddings', str(self.train_path),
                                        '--test-embeddings', str(self.test_path), '--parameters', str(args.parameters),
                                        '--reference-sets', 'Both', '--selection', str(selection_path),
                                        '--output', str(self.root / 'selected_test')])
        run(test_args)
        summary = json.loads((test_args.output / 'summary.json').read_text())
        self.assertEqual({r['model'] for r in summary}, {'selma', 'balanced'})
        self.assertTrue(list(test_args.output.rglob('prediction_ranks.csv')))
        with self.assertRaisesRegex(ValueError, 'test results cannot select'):
            freeze(test_args.output, self.root / 'invalid_selection.json')
        # A frozen configuration cannot silently change on the test run.
        configs = json.loads(args.parameters.read_text())
        configs['balanced']['centroid_weight'] = .9
        args.parameters.write_text(json.dumps(configs))
        with self.assertRaisesRegex(ValueError, 'Configuration changed'):
            preflight(test_args)

    def test_validation_files_cannot_change_after_preparation(self):
        data = self.root / 'validation_data'
        self.prepare_validation(data)
        target = pd.read_csv(data / 'validation/CrossNews.csv')
        target.iloc[0, target.columns.get_loc('genre')] = 'Tweet'
        target.to_csv(data / 'validation/CrossNews.csv', index=False)
        # Expected failure occurs before embedding coverage or scoring.
        args = parser().parse_args(['--data-dir', str(data), '--train-embeddings', str(self.train_path),
                                    '--test-embeddings', str(self.test_path), '--split', 'validation'])
        with self.assertRaisesRegex(ValueError, 'changed after preparation'):
            preflight(args)

    def test_selection_rejects_validation_test_author_overlap(self):
        args, selection_path, selection = self.validation_run()
        target = pd.read_csv(self.data / 'test/CrossNews.csv')
        target['author'] = selection['validation_authors'][0]
        target.to_csv(self.data / 'test/CrossNews.csv', index=False)
        test_args = parser().parse_args(['--data-dir', str(self.data), '--train-embeddings', str(self.train_path),
                                        '--test-embeddings', str(self.test_path), '--parameters', str(args.parameters),
                                        '--reference-sets', 'Both', '--selection', str(selection_path)])
        with self.assertRaisesRegex(ValueError, 'author sets overlap'):
            preflight(test_args)

    def test_embedding_combiner_respects_custom_output_directory(self):
        output = self.root / 'custom_embeddings'
        (output / 'train_partitions').mkdir(parents=True)
        (output / 'train_partitions/partition_0_1.json').write_text(json.dumps({'001': [1., 2.]}))
        script = Path(__file__).resolve().parents[1] / 'src/generate_selma_embeddings.py'
        result = subprocess.run([sys.executable, str(script), 'train', 'combine', '--output-dir', str(output)],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((output / 'train.json').read_text()), {'001': [1., 2.]})


if __name__ == '__main__':
    unittest.main()
