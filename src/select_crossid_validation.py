"""Freeze CROSS-ID configurations from a completed silver validation run."""
import argparse
import hashlib
import json
from pathlib import Path


def freeze(folder, output):
    manifest_path, summary_path = folder / 'manifest.json', folder / 'summary.json'
    manifest = json.loads(manifest_path.read_text())
    rows = json.loads(summary_path.read_text())
    if manifest.get('status') != 'completed' or manifest.get('split_role') != 'validation':
        raise ValueError('Selection requires a completed validation run; gold test results cannot select configurations.')
    if manifest.get('validation_protocol', {}).get('role') != 'silver_validation':
        raise ValueError('Selection requires a verified silver validation protocol.')
    selected = {}
    for reference in manifest['counts']['references']:
        candidates = [r for r in rows if r['references'] == reference and r['targets'] == 'Overall'
                      and r['model'] in manifest['parameters']]
        if not candidates:
            raise ValueError(f'No CROSS-ID validation candidates for {reference}.')
        # Primary metric fixed before evaluation: overall top-1 accuracy.
        # Equal genre counts are required by the benchmark validation preflight.
        winner = sorted(candidates, key=lambda r: (-r['Accuracy'], -r['Mean_Reciprical_Rank'], r['model']))[0]
        selected[reference] = {'model': winner['model'], 'parameters': manifest['parameters'][winner['model']],
                               'validation_accuracy': winner['Accuracy'],
                               'validation_mrr': winner['Mean_Reciprical_Rank']}
    result = {
        'format': 'crossid_validation_selection_v1',
        'selection_metric': 'Overall Accuracy; ties: overall MRR, then configuration name',
        'selected': selected, 'validation_authors': manifest['author_labels'],
        'crossid_source_sha256': manifest['source_sha256']['src/attribution_models/crossid.py'],
        'validation_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'validation_summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        'validation_protocol': manifest['validation_protocol'],
    }
    with output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(selected, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation-results', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    try:
        freeze(args.validation_results, args.output)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        p.error(str(exc))
