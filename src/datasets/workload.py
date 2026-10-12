"""Independent source adapters and deterministic temporal resampling."""
import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path

FIELDS = ('request_id', 'intent_id', 'query', 'topic_id', 'phase_id', 'step')


def primary_rows(rows):
    result = []
    for row in rows:
        if not {'id', 'prompt', 'ID_Set'} <= row.keys():
            raise ValueError('Expected SemCacheLMArena columns id, prompt, ID_Set')
        if row['ID_Set'] is None or not isinstance(row['prompt'], str) or not row['prompt'].strip():
            raise ValueError('Primary rows require class labels and nonempty prompts')
        result.append({'query': row['prompt'], 'intent_id': str(row['ID_Set']), 'topic_id': None})
    return result


def load_primary(revision=None):
    from datasets import load_dataset
    return primary_rows(load_dataset('vCache/SemBenchmarkLmArena', split='train', revision=revision))


def generate(rows, phases=None, seed=0):
    """phases: ordered dictionaries mapping intent IDs to exact request counts."""
    groups = defaultdict(list)
    for row in rows:
        if row.get('intent_id') is None:
            raise ValueError('Temporal intent resampling requires source labels')
        groups[str(row['intent_id'])].append(row)
    if phases is None:
        ids = sorted(groups)
        if len(ids) < 3:
            raise ValueError('Default phases require at least three source classes')
        a, b, c = ids[:3]
        phases = [{a: 24, b: 8}, {a: 8, b: 24}, {b: 8, c: 24}]
    rng = random.Random(seed)
    output = []
    for phase_id, counts in enumerate(phases):
        requests = []
        for intent, count in counts.items():
            intent = str(intent)
            if intent not in groups or not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError('Each phase count must be positive and refer to an available class')
            variants = list(groups[intent])
            rng.shuffle(variants)
            for i in range(count):
                requests.append(variants[i % len(variants)])
        rng.shuffle(requests)
        for row in requests:
            step = len(output)
            output.append(dict(request_id=f's{seed}-r{step}', intent_id=str(row['intent_id']),
                               query=row['query'], topic_id=row.get('topic_id'), phase_id=phase_id, step=step))
    validate(output, require_labels=True, require_drift=True)
    return output


def validate(rows, require_labels=False, require_drift=False):
    if not rows:
        raise ValueError('Workload must be nonempty')
    seen, phases = set(), []
    distributions = defaultdict(Counter)
    for step, row in enumerate(rows):
        if not set(FIELDS) <= row.keys() or row['step'] != step or row['request_id'] in seen:
            raise ValueError('Invalid schema, chronology, or duplicate request ID')
        if not isinstance(row['query'], str) or not row['query'].strip():
            raise ValueError('Queries must be nonempty strings')
        if require_labels and row['intent_id'] is None:
            raise ValueError('Missing intent label')
        phase = row['phase_id']
        if not phases or phase != phases[-1]:
            if phase in phases:
                raise ValueError('Phase order must be contiguous')
            phases.append(phase)
        seen.add(row['request_id'])
        distributions[phase][row['intent_id']] += 1
    normalized = [{k: v / sum(c.values()) for k, v in c.items()} for c in distributions.values()]
    if require_drift and (len(normalized) < 2 or any(a == b for a, b in zip(normalized, normalized[1:]))):
        raise ValueError('Adjacent phase intent distributions must differ')
    return normalized


def wildchat_rows(rows, phase_size=100):
    """First user turn only; chronological conversation end time, no equivalence labels."""
    if phase_size < 1:
        raise ValueError('phase_size must be positive')
    result = []
    for row in sorted(rows, key=lambda r: (r['timestamp'], r['conversation_id'])):
        query = next((t['content'] for t in row['conversation'] if t['role'] == 'user' and t['content'].strip()), None)
        if query is not None:
            i = len(result)
            result.append(dict(request_id=str(row['conversation_id']), intent_id=None, query=query,
                               topic_id=None, phase_id=i // phase_size, step=i))
    validate(result)
    return result


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, rows):
    Path(path).write_text(''.join(json.dumps(row) + '\n' for row in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', help='JSON ordered list of intent-to-count maps')
    parser.add_argument('--revision', help='Dataset commit revision for reproducible downloads')
    args = parser.parse_args()
    phases = json.loads(Path(args.config).read_text()) if args.config else None
    write(args.output, generate(load_primary(args.revision), phases, args.seed))
