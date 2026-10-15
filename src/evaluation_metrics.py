"""Method-independent metrics for served semantic-cache responses."""


def reuse_metrics(records):
    """Summarize status-bearing records with explicit quality denominators."""
    n = len(records)
    if not n:
        raise ValueError('Cannot summarize empty reuse records')
    correct = sum(r['status'] == 'correct' for r in records)
    incorrect = sum(r['status'] == 'incorrect' for r in records)
    served = sum(r['status'] != 'not_reused' for r in records)
    evaluable_served = correct + incorrect
    return dict(served_hit_rate=served/n, quality_adjusted_hit_rate=correct/n,
                reuse_precision=correct/evaluable_served if evaluable_served else None,
                incorrect_share_of_evaluable_reuse=incorrect/evaluable_served if evaluable_served else None,
                unevaluable_share_of_served_hits=(served-evaluable_served)/served if served else None)
