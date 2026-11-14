"""Seeded duplicate-deal evaluation against the pinned Fell Omen 2 opponent."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import platform
import subprocess
import time
from importlib.metadata import version

import numpy as np
from scipy.stats import t as student_t

from .checkpointing import LoadedHULHEPolicy, sha256_file
from .fell_omen2 import FellOmen2Policy, validate_probabilities
from .fo2_setup import DEFAULT_ASSETS, asset_manifest, bridge_fingerprint
from .game import load_hulhe_game, serialisable_game_definition, SMALL_BET_CHIPS


class AlwaysCallPolicy:
    def action_probabilities(self, state):
        return {a: float(a == 1) for a in state.legal_actions()}


class UniformPolicy:
    def action_probabilities(self, state):
        actions = state.legal_actions()
        return {a: 1 / len(actions) for a in actions}


def play_hand(game, policies, deal, *, seed, pair, leg):
    """Use fixed chance cards and independent local action RNGs for each seat."""
    if len(deal) != 9 or len(set(deal)) != 9 or any(not 0 <= c < 52 for c in deal):
        raise ValueError('A deal must contain nine distinct card ids in [0, 51]')
    state = game.new_initial_state()
    rng = [np.random.default_rng(np.random.SeedSequence([seed, pair, leg, seat])) for seat in (0, 1)]
    chance = iter(deal)
    actions = []
    while not state.is_terminal():
        if state.is_chance_node():
            card = next(chance)
            if card not in dict(state.chance_outcomes()):
                raise ValueError('Deal disagrees with OpenSpiel chance outcomes')
            state.apply_action(card)
            continue
        player = state.current_player()
        policy = policies[player]
        probabilities = policy.action_probabilities(state)
        if set(probabilities) != set(state.legal_actions()):
            raise ValueError('Policy must return exactly the legal action ids')
        dense = [probabilities.get(a, 0) for a in range(3)]
        probabilities = validate_probabilities(dense, state.legal_actions())
        legal = list(probabilities)
        action = int(rng[player].choice(legal, p=[probabilities[a] for a in legal]))
        actions.append({'player': player, 'action': action, 'legal': legal,
                        'probabilities': probabilities,
                        'opponent_lookup': getattr(policy, 'last_diagnostics', None)})
        state.apply_action(action)
    returns = state.returns()
    if sum(returns) != 0:
        raise AssertionError('Non-zero-sum game settlement')
    return {'pair': pair, 'leg': leg, 'deal': list(map(int, deal)),
            'actions': actions, 'returns_chips': returns}


def run_matches(candidate, opponent, *, pairs, seed, output, metadata=None):
    if pairs < 1 or seed < 0:
        raise ValueError('pairs must be positive and seed nonnegative')
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Evaluation requires an empty output directory')
    root.mkdir(parents=True, exist_ok=True)
    manifest = {'format_version': 1, 'game': serialisable_game_definition(),
                'seed': seed, 'pairs': pairs, 'hands': 2 * pairs,
                'deal_scheme': 'same nine cards in both legs; candidate seat 0 then seat 1',
                'metadata': metadata or {}}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    started = time.monotonic()
    deal_rng = np.random.default_rng(np.random.SeedSequence([seed, 314159]))
    scores = []
    game = load_hulhe_game()
    completed = 0
    try:
        with (root / 'hands.jsonl').open('w') as log:
            for pair in range(pairs):
                deal = deal_rng.choice(52, size=9, replace=False).tolist()
                pair_returns = []
                for leg in (0, 1):
                    policies = [candidate, opponent] if leg == 0 else [opponent, candidate]
                    record = play_hand(game, policies, deal, seed=seed, pair=pair, leg=leg)
                    record['candidate_seat'] = leg
                    record['candidate_return_chips'] = record['returns_chips'][leg]
                    pair_returns.append(record['candidate_return_chips'])
                    log.write(json.dumps(record, sort_keys=True) + '\n')
                    log.flush()
                    completed += 1
                scores.append(sum(pair_returns) / 2 / SMALL_BET_CHIPS)
    except BaseException as exc:
        (root / 'failure.json').write_text(json.dumps({'completed_hands': completed,
            'error': f'{type(exc).__name__}: {exc}'}, indent=2) + '\n')
        raise
    mean = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(pairs)) if pairs > 1 else None
    margin = float(student_t.ppf(.975, pairs - 1) * se) if se is not None else None
    summary = {'pairs': pairs, 'hands': completed, 'candidate_mean_small_bets_per_hand': mean,
               'paired_standard_error': se,
               'approximate_paired_t_interval_95': None if margin is None else [mean - margin, mean + margin],
               'uncertainty_note': 'Pairs are the independent units; a small smoke match is not evidence of playing strength.',
               'elapsed_seconds': time.monotonic() - started}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument('--checkpoint', type=Path)
    candidate.add_argument('--baseline', choices=['call', 'random'])
    parser.add_argument('--opponent', choices=['fo2', 'call', 'random'], default='fo2')
    parser.add_argument('--assets', type=Path, default=DEFAULT_ASSETS)
    parser.add_argument('--pairs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=60, help='Seconds per FO2 request')
    parser.add_argument('--fo2-free-fold', choices=['error', 'check'], default='error',
                        help='Default: reject incompatible free folds. check labels an adapted opponent.')
    args = parser.parse_args(argv)
    if args.pairs < 1 or args.seed < 0 or args.timeout <= 0:
        parser.error('pairs/timeout must be positive; seed must be nonnegative')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use an empty output directory')
    game = load_hulhe_game()
    metadata = {'python': platform.python_version(), 'numpy': np.__version__,
                'open_spiel': version('open_spiel'), 'opponent': args.opponent}
    if args.checkpoint:
        import torch
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            learner = LoadedHULHEPolicy(game, args.checkpoint)
        metadata['candidate'] = {'checkpoint': str(args.checkpoint.resolve()),
                                 'sha256': sha256_file(args.checkpoint), 'torch': torch.__version__}
    else:
        learner = AlwaysCallPolicy() if args.baseline == 'call' else UniformPolicy()
        metadata['candidate'] = {'baseline': args.baseline}
    if args.opponent == 'fo2':
        context = FellOmen2Policy(game, args.assets, timeout=args.timeout, free_fold=args.fo2_free_fold)
        metadata['fo2'] = {'upstream_commit': asset_manifest()['commit'],
                           'free_fold_rule': args.fo2_free_fold,
                           'policy_variant': 'strict' if args.fo2_free_fold == 'error' else 'free-fold-to-check adaptation',
                           'bridge_fingerprint': bridge_fingerprint(),
                           'build': context.build,
                           'java': subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT, text=True)}
    else:
        context = nullcontext(AlwaysCallPolicy() if args.opponent == 'call' else UniformPolicy())
    with context as opponent:
        result = run_matches(learner, opponent, pairs=args.pairs, seed=args.seed,
                             output=args.output, metadata=metadata)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
