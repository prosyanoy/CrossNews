"""CPU regression tests, including real optimizer steps with a tiny local Transformer."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

try:
    import torch
    import ijson
    import transformers
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False

if LEARNING_AVAILABLE:
    from prepare_crossid_learning import prepare
    from train_crossid_phase2 import train as train_phase2, parser as phase2_parser
    from train_crossid_phase3 import train as train_phase3, calibrate, parser as phase3_parser
    from benchmark_crossid_phase23 import benchmark, parser as benchmark_parser, frozen_scores
    from crossid_learning.common import (load_embedding_view, load_protocol, protect_evaluation,
                                         read_documents, text_hash, check_reference_target)
    from crossid_learning.fusion import apply_fusion, fit_fusion
    from crossid_learning.models import CrossIDAdapter, GradientReverse, load_adapter, training_loss
    from crossid_learning.pipeline import validate_fusion
    from crossid_learning.reranking import (FEATURE_NAMES, build_candidates, candidate_features, stylometry)
    from crossid_learning.retrieval import (RetrievalIndex, descending_order, prediction_frame, reranked_order)
    from crossid_learning.sampling import BundleSampler


def corpus():
    return [dict(id=f'{a}-{genre}-{i}', author=f'author{a}', genre=genre,
                 text=f'Topic {"alpha" if i%2 else "beta"} report number {a*100+i} '
                      f'{genre} marker {a*1000+i}.' + ('!'*a))
            for a in range(12) for genre in ['Article', 'Tweet'] for i in range(6)]


@unittest.skipUnless(LEARNING_AVAILABLE, 'Install requirements-crossid-learning.txt for learning tests.')
class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare_fixture(self, reverse=False, output='data'):
        silver = self.folder / ('reverse.json' if reverse else 'silver.json')
        silver.write_text(json.dumps(corpus()[::-1] if reverse else corpus()))
        gold = self.folder / 'gold.json'
        gold.write_text(json.dumps([dict(id='gold1', author='gold_author', genre='Article', text='Unique gold article.')]))
        path = self.folder / output
        prepare(silver, gold, path, 2, 2, 2, 2, 1, 6, seed=13)
        return path

    def test_reverse_gradient_and_bundle_padding_permutation(self):
        x = torch.tensor([1., 2.], requires_grad=True)
        GradientReverse.apply(x, .4).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([-.4, -.4]))
        model = CrossIDAdapter(4, 8, 4, 2, 2)
        values = model.encode(torch.randn(1, 3, 4))
        mask = torch.tensor([[True, True, False]])
        p = model.bundle(values, mask)
        permuted = model.bundle(values[:, [1, 2, 0]], mask[:, [1, 2, 0]])
        torch.testing.assert_close(p, permuted)
        values[:, 2] = 1000
        torch.testing.assert_close(p, model.bundle(values, mask))
        with self.assertRaises(ValueError):
            model.bundle(values, torch.zeros_like(mask))

    def test_contrastive_optimizer_updates_adapter_slots_and_heads(self):
        model = CrossIDAdapter(4, 8, 4, 2, 2)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        loss, _ = training_loss(model, torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 3, 4),
                                torch.ones(3, 3, dtype=torch.bool), torch.tensor([0,1,0,1,0,1]),
                                torch.tensor([0,0,0,1,1,1]))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        for name in ['slots', 'skip.weight', 'adapter.0.weight', 'genre_head.weight', 'topic_head.weight']:
            self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]), name)

    def test_protocol_is_order_independent_disjoint_and_hash_checked(self):
        data = self.prepare_fixture()
        reverse = self.prepare_fixture(True, 'reversed_data')
        p, _ = load_protocol(data); q, _ = load_protocol(reverse)
        self.assertEqual(p['author_splits'], q['author_splits'])
        self.assertEqual(p['files'], q['files'])
        roles = [set(v) for v in p['author_splits'].values()]
        self.assertEqual(sum(map(len, roles)), len(set.union(*roles)))
        frames = [read_documents(data/role/('documents.csv' if role=='train' else 'targets.csv'))
                  for role in p['author_splits']]
        for i, a in enumerate(frames):
            for b in frames[i+1:]:
                self.assertFalse(set(a.id) & set(b.id))
                self.assertFalse(set(a.text.map(text_hash)) & set(b.text.map(text_hash)))
        (data/'dev/targets.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'split changed'):
            load_protocol(data)

    def test_crossgenre_bundles_and_topic_negative_sampling(self):
        frame = pd.DataFrame(corpus()).reset_index(drop=True)
        topics = np.arange(len(frame)) % 2
        sampler = BundleSampler(frame, topics, 13)
        for _ in range(10):
            anchors, positives, bundles = sampler.sample(4, 4)
            self.assertEqual(len(set(frame.iloc[anchors].author)), 4)
            for anchor, positive, bundle in zip(anchors, positives, bundles):
                self.assertEqual(frame.iloc[anchor].author, frame.iloc[positive].author)
                self.assertNotEqual(frame.iloc[anchor].genre, frame.iloc[positive].genre)
                self.assertTrue(all(frame.iloc[ids].author == frame.iloc[anchor].author for ids in bundle))
                self.assertNotIn(anchor, bundle)

    def test_missing_view_and_evaluation_leakage_rejected(self):
        data = self.prepare_fixture(); p, _ = load_protocol(data)
        refs = read_documents(data/'test/query/CrossNews_Both.csv')
        targets = read_documents(data/'test/targets.csv')
        metadata = {'train_authors': p['author_splits']['train'], 'dev_authors': p['author_splits']['dev']}
        protect_evaluation(refs, targets, metadata)
        metadata['train_authors'].append(refs.author.iloc[0])
        with self.assertRaisesRegex(ValueError, 'authors overlap'):
            protect_evaluation(refs, targets, metadata)
        path = self.folder/'embeddings.json'; path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Missing'):
            load_embedding_view([path], refs)

    def test_no_oracle_candidate_injection_and_full_tail_ranks(self):
        refs = pd.DataFrame([dict(id=f'r{i}', author=f'a{i}', genre='Article', text=f'Reference text {i}.') for i in range(3)])
        targets = pd.DataFrame([dict(id='q', author='a2', genre='Tweet', text='Target text.')])
        adapter = CrossIDAdapter(3, 4, 3, 2, 2)
        index = RetrievalIndex(adapter, refs, np.eye(3, dtype=np.float32))
        scores = np.asarray([[.9, .8, .1]])
        test = build_candidates(index, targets, np.asarray([[1.,0,0]]), scores, scores, scores, 1, 1)
        self.assertEqual(test.candidates.tolist(), [[0]])
        self.assertEqual(test.labels.tolist(), [0])
        training = build_candidates(index, targets, np.asarray([[1.,0,0]]), scores, scores, scores, 1, 1, True)
        self.assertEqual(training.candidates.tolist(), [[2]])
        order = reranked_order(scores, test.candidates, np.asarray([[100.]]))
        self.assertEqual(prediction_frame(targets, index.authors, order)['rank'].tolist(), [3])
        self.assertNotIn('a2', test.pairs[0][0]); self.assertNotIn('a0', test.pairs[0][1])
        relabeled = targets.assign(author='a0')
        other = build_candidates(index, relabeled, np.asarray([[1.,0,0]]), scores, scores, scores, 1, 1)
        np.testing.assert_equal(test.candidates, other.candidates)
        np.testing.assert_equal(test.base_features, other.base_features)
        self.assertEqual(test.pairs, other.pairs)
        self.assertEqual(other.labels.tolist(), [1])

    def test_existing_text_overlap_option_keeps_id_leakage_guard(self):
        refs = pd.DataFrame([dict(id='r1',author='a',genre='Article',text='Same text.'),
                             dict(id='r2',author='b',genre='Article',text='Other text.')])
        targets = pd.DataFrame([dict(id='t1',author='a',genre='Tweet',text=' same   TEXT. '),
                                dict(id='t2',author='b',genre='Tweet',text='Fresh text.')])
        with self.assertRaisesRegex(ValueError,'text overlaps'):
            check_reference_target(refs, targets)
        check_reference_target(refs, targets, allow_text_overlap=True)
        targets.loc[0,'id'] = 'r1'
        with self.assertRaisesRegex(ValueError,'IDs overlap'):
            check_reference_target(refs, targets, allow_text_overlap=True)

    def test_stylometry_mean_logits_and_numeric_fusion_roundtrip(self):
        self.assertTrue(np.isfinite(stylometry('')).all())
        features = np.random.default_rng(13).normal(size=(8,2,len(FEATURE_NAMES)))
        labels = np.tile([1,0], (8,1))
        artifact = fit_fusion(features, labels, {'status': 'completed'})
        restored = json.loads(json.dumps(artifact))
        np.testing.assert_allclose(apply_fusion(features, artifact), apply_fusion(features, restored))
        artifact['feature_names'][0] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'schema'):
            apply_fusion(features, artifact)

    def test_frozen_selma_matches_upstream_formulation(self):
        from scipy.spatial.distance import cosine
        refs = pd.DataFrame({'author':['a','a','b','b']})
        reference = np.asarray([[1.,2,3], [3.,1,2], [2.,4,1], [1.,5,1]])
        target = np.asarray([[2.,1,3], [3.,4,1]])
        actual, _ = frozen_scores(refs, reference, target, ['a','b'])
        expected = [[-round(cosine(reference[np.asarray(refs.author==a)].mean(axis=0), t),4)
                     for a in ['a','b']] for t in target]
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_full_training_calibration_and_benchmark_cpu_workflow(self):
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
        data = self.prepare_fixture(); p, _ = load_protocol(data)
        documents = pd.DataFrame(corpus())
        rng = np.random.default_rng(13)
        author_vectors = {a: rng.normal(size=6) for a in documents.author.unique()}
        reference = {d.id: (author_vectors[d.author] + rng.normal(scale=.03,size=6)).tolist()
                     for _,d in documents.iterrows()}
        target = {i: (np.asarray(v)+.02).tolist() for i,v in reference.items()}
        ref_path, target_path = self.folder/'ref.json', self.folder/'target.json'
        ref_path.write_text(json.dumps(reference)); target_path.write_text(json.dumps(target))
        phase2 = self.folder/'phase2'
        shared = ['--data-dir', str(data), '--reference-embeddings', str(ref_path),
                  '--target-embeddings', str(target_path), '--device', 'cpu']
        train_phase2(phase2_parser().parse_args(shared+['--output',str(phase2),'--epochs','2',
                     '--steps-per-epoch','3','--authors-per-batch','4','--hidden-dim','8',
                     '--output-dim','6','--prototypes','2','--topic-clusters','2']))
        checkpoint = phase2/'adapter.pt'
        adapter, meta = load_adapter(checkpoint)
        self.assertEqual(meta['status'], 'completed')
        base = self.folder/'tiny_bert'; base.mkdir()
        (base/'vocab.txt').write_text('\n'.join(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]',
                                                'topic','alpha','beta','report','number','article',
                                                'tweet','marker','.','!']+[str(i) for i in range(1200)]))
        tokenizer = BertTokenizerFast(vocab_file=str(base/'vocab.txt'))
        tokenizer.save_pretrained(base)
        tiny = BertForSequenceClassification(BertConfig(vocab_size=len(tokenizer), hidden_size=16,
                num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=1))
        tiny.save_pretrained(base, safe_serialization=True)
        phase3 = self.folder/'phase3'
        train_phase3(phase3_parser().parse_args(['train']+shared+['--phase2',str(checkpoint),
                     '--model',str(base),'--output',str(phase3),'--epochs','2','--max-pairs','24',
                     '--dev-queries','4','--max-length','48','--top-candidates','2',
                     '--references-per-candidate','1','--batch-size','4']))
        fusion = phase3/'fusion.json'
        calibrate(phase3_parser().parse_args(['calibrate']+shared+['--phase2',str(checkpoint),
                  '--phase3',str(phase3),'--output',str(fusion),'--top-candidates','2',
                  '--references-per-candidate','1','--batch-size','4']))
        out = self.folder/'benchmark'
        bench_args = benchmark_parser().parse_args(['--data-dir',str(data/'test'),
                     '--reference-embeddings',str(ref_path),'--target-embeddings',str(target_path),
                     '--phase2',str(checkpoint),'--phase3',str(phase3),'--fusion',str(fusion),
                     '--output',str(out),'--top-candidates','2','--references-per-candidate','1',
                     '--batch-size','4'])
        checkpoint_before = checkpoint.read_bytes()
        benchmark(bench_args)
        self.assertEqual(checkpoint_before, checkpoint.read_bytes())
        rows = pd.read_csv(out/'summary.csv')
        self.assertEqual(len(rows), 18)
        self.assertEqual(set(rows.model), {'selma','frozen_reference_only','phase2_reference_only',
                                         'phase2_learned_profiles','phase3_cross_encoder','phase3_fusion'})
        for path in out.rglob('prediction_ranks.csv'):
            predictions = pd.read_csv(path)
            self.assertEqual(len(predictions),4)
            self.assertTrue(((predictions.prediction==predictions.label)==(predictions['rank']==1)).all())
        bench_args.top_candidates = 1
        with self.assertRaisesRegex(ValueError, 'budgets'):
            benchmark(bench_args)
        artifact = json.loads(fusion.read_text())
        artifact['metadata']['phase2_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            validate_fusion(artifact, checkpoint, phase3, artifact['metadata']['routing'])


if __name__ == '__main__':
    unittest.main()
