"""Exact river constraints and explicitly unscored preflop comparisons."""
from itertools import combinations
import numpy as np
import torch
from .game import load_hulhe_game
from .card_features import rank_cards
from .features import StructuredHULHEMLP


def positions():
    game = load_hulhe_game()
    result = []
    for hero in (0, 1):
        for kind in ('private_royal', 'board_royal'):
            hole = [48, 44] if kind == 'private_royal' else [0, 5]
            board = [40, 36, 32, 1, 6] if kind == 'private_royal' else [32, 36, 40, 44, 48]
            other = [c for c in range(52) if c not in hole + board][:2]
            deck = iter((hole + other if hero == 0 else other + hole) + board)
            state = game.new_initial_state()
            dealt = 0
            while dealt < 9:
                if state.is_chance_node():
                    state.apply_action(next(deck)); dealt += 1
                else:
                    state.apply_action(1)
            # River first actor is P1. Check once to reverse the betting sequence.
            if hero == 0:
                state.apply_action(1)
            while 2 in state.legal_actions():
                state.apply_action(2)
            assert state.current_player() == hero and state.legal_actions() == [0, 1]
            rank = int(rank_cards([hole + board])[0][0])
            remaining = [c for c in range(52) if c not in hole + board]
            ranks = rank_cards([list(pair) + board for pair in combinations(remaining, 2)])[0]
            assert np.all(rank >= ranks)
            # A capped river call ends the hand. All possible opponent holes lose
            # (private nuts) or tie (board nuts), so calling strictly dominates folding.
            call, fold = state.child(1), state.child(0)
            assert call.is_terminal() and fold.is_terminal()
            assert call.returns()[hero] > fold.returns()[hero]
            result.append(dict(name=f'{kind}_p{hero}_facing_cap', state=state,
                exact=True, expected_action=1, reason='Capped river: guaranteed win/tie; call strictly dominates fold',
                opponent_combinations=len(ranks)))
    for name, cards in [('AA', [48,49,0,5]), ('72o', [20,1,48,49])]:
        state = game.new_initial_state()
        for c in cards: state.apply_action(c)
        result.append(dict(name=f'preflop_{name}', state=state, exact=False,
                           reason='Context only; no exact equilibrium action asserted'))
    return result


def regret_models(metadata, encoder):
    config = metadata['config']
    models = []
    for row in metadata['regret_trainers']:
        model = StructuredHULHEMLP(encoder.policy_layout,
            [config['num_hiddens']] * config['num_layers'], 3,
            branch_width=config['structured_branch_width'])
        model.load_state_dict(row['model']); model.eval(); models.append(model)
    return models


def compare_probes(data, encoder, metadata, models):
    probes = positions()
    regrets = regret_models(metadata, encoder)
    encoded = np.stack([encoder.information_state(p['state'], p['state'].current_player()) for p in probes])
    matches = {i: [] for i in range(len(probes))}
    for start in range(0, len(data[0]), 4096):
        x = data[0].numpy_batch(start, min(start+4096, len(data[0])))
        for i, feature in enumerate(encoded):
            matches[i].extend((start + np.flatnonzero(np.all(x == feature, axis=1))).tolist())
    result = []
    with torch.no_grad():
        for i, p in enumerate(probes):
            state = p['state']; mask = np.asarray(state.legal_actions_mask(), bool)
            x = torch.from_numpy(encoded[i]); player = state.current_player()
            raw = regrets[player](x).numpy()
            from vr_deep_cfr.solver import RegretTrainer
            from types import SimpleNamespace
            settings = SimpleNamespace(use_regret_matching_argmax=metadata['config']['use_regret_matching_argmax'],
                uniform_regret_ties=metadata['config']['uniform_regret_ties'])
            strategy = RegretTrainer.regret_matching(settings, raw, state.legal_actions())
            row = {k:v for k,v in p.items() if k != 'state'}
            row.update(history=state.history(), legal_mask=mask.tolist(), regret_values=raw.tolist(),
                       current_regret_strategy=strategy.tolist(), replay_present=bool(matches[i]),
                       replay_target=data[1][matches[i][0]].tolist() if matches[i] else None,
                       policies={})
            for name, model in models.items():
                probs = torch.softmax(model(x).masked_fill(~torch.tensor(mask), -1e20), -1).tolist()
                row['policies'][name] = dict(probabilities=probs,
                    forbidden_probability=probs[0] if p['exact'] else None)
            result.append(row)
    return dict(positions=result, caveat='Current regret strategy is instantaneous; replay target is a historical weighted average. Differences alone are not errors. Missing replay targets are unavailable, never imputed.')
