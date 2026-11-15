"""Sanitized OpenSpiel observations and a persistent local Fell Omen 2 process."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time

import numpy as np

from .fo2_setup import DEFAULT_ASSETS, verify_build
from .game import validate_hulhe_game

# Terminal-node numbering used by FellOmen_2.nodeToTerminalNodeIndex.
_TERMINAL = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8}


def card_string(card):
    if not 0 <= card < 52:
        raise ValueError('Invalid OpenSpiel card')
    return '23456789TJQKA'[card // 4] + 'cdhs'[card % 4]


def action_node(street, actor, raises):
    first = 0 if street == 0 else 1
    # The small blind/button is p2 in FO2; the big blind is p1.
    return ([5, 9, 3, 7, 1] if actor == first else [10, 4, 8, 2, 6])[raises]


def observation_request(state):
    """Only own cards, revealed board and public betting events cross the bridge."""
    if state.is_terminal() or state.is_chance_node():
        raise ValueError('Fell Omen 2 accepts decision states only')
    player = state.current_player()
    raw = state.information_state_tensor(player)
    hole = np.flatnonzero(np.asarray(raw[2:54]))
    if len(hole) != 2:
        raise ValueError('Expected two visible private cards')
    board, histories = [], [-1, -1, -1, -1]
    events = [[], [], [], []]
    dealt = street = raises = 0
    for event in state.full_history():
        if event.player == -1:
            dealt += 1
            if dealt <= 4:
                continue  # Never read or transmit the other player's hole-card ids.
            board.append(card_string(event.action))
            next_street = 1 if len(board) <= 3 else len(board) - 2
            if next_street != street:
                street, raises = next_street, 0
        else:
            if event.action == 1:
                histories[street] = _TERMINAL.get(action_node(street, event.player, raises), 8)
            elif event.action == 2:
                raises += 1
            else:
                raise ValueError('A live decision history cannot contain a fold or all-in')
            events[street].append('c' if event.action == 1 else 'r')
    if len(board) not in (0, 3, 4, 5) or any(h < 0 for h in histories[:street]):
        raise ValueError('Incomplete public history')
    return {'hole': ' '.join(card_string(int(c)) for c in hole), 'board': ' '.join(board),
            'street': street, 'actor': player, 'raises': raises,
            'preflop': histories[0] if street > 0 else -1,
            'flop': histories[1] if street > 1 else -1,
            'turn': histories[2] if street > 2 else -1,
            'history': '/'.join(''.join(a) for a in events[:street + 1])}


def validate_probabilities(values, legal, *, tolerance=2e-6):
    """Only documented decimal rounding may be corrected; never replace strategy."""
    p = np.asarray(values, dtype=np.float64)
    # Moving free-fold mass into check can put a rounded table entry just
    # above one. Apply the same decimal tolerance as the total-mass check;
    # normalization below preserves the relative legal-action probabilities.
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1 + tolerance):
        raise ValueError('Invalid FO2 probability vector')
    legal = list(legal)
    if not legal or any(a not in (0, 1, 2) for a in legal):
        raise ValueError('Invalid legal action set')
    if abs(p.sum() - 1) > tolerance:
        raise ValueError(f'FO2 probabilities do not sum to one: {p.sum()}')
    illegal = [a for a in range(3) if a not in legal]
    if p[illegal].sum() > tolerance:
        raise ValueError(f'FO2 assigned probability to illegal actions: {p.tolist()}, legal={legal}')
    p[illegal] = 0
    if p.sum() <= 0:
        raise ValueError('FO2 has no legal probability mass')
    p /= p.sum()
    return {a: float(p[a]) for a in legal}


class FellOmen2Policy:
    """OpenSpiel-style policy; one sequential request stream per instance."""
    def __init__(self, game, assets=DEFAULT_ASSETS, *, timeout=60, free_fold='error'):
        validate_hulhe_game(game)
        self.game = game
        self.assets = Path(assets).resolve()
        self.build = verify_build(self.assets)
        if timeout <= 0:
            raise ValueError('timeout must be positive')
        self.timeout = timeout
        if free_fold not in ('error', 'check'):
            raise ValueError('free_fold must be error or check')
        self.free_fold = free_fold
        self.last_diagnostics = None
        self._errors = tempfile.TemporaryFile()
        self._buffer = b''
        self._selector = selectors.DefaultSelector()
        self._process = None
        try:
            self._process = subprocess.Popen([
                'java', '-Xmx256m', '-Djava.library.path=' + str(self.assets / 'no-native-libraries'),
                '-cp', os.pathsep.join([str(self.assets / 'classes'), str(self.assets / 'meerkat.jar')]),
                'com.ibot.FellOmenBridge'], cwd=self.assets,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._errors)
            self._selector.register(self._process.stdout, selectors.EVENT_READ)
        except BaseException:
            self.close()
            raise

    def request(self, observation, *, reference=False):
        if self._process is None:
            raise RuntimeError('FO2 process is closed')
        request = dict(observation, mode='reference' if reference else 'query')
        try:
            self._process.stdin.write((json.dumps(request, separators=(',', ':')) + '\n').encode())
            self._process.stdin.flush()
            deadline = time.monotonic() + self.timeout
            while b'\n' not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._selector.select(remaining):
                    raise TimeoutError('Fell Omen 2 decision timed out')
                data = os.read(self._process.stdout.fileno(), 65536)
                if not data:
                    self._errors.seek(0)
                    raise RuntimeError('FO2 process exited: ' + self._errors.read().decode(errors='replace')[-2000:])
                self._buffer += data
                if len(self._buffer) > 65536:
                    raise RuntimeError('Oversized FO2 response')
            line, self._buffer = self._buffer.split(b'\n', 1)
            result = json.loads(line)
            if 'error' in result:
                raise RuntimeError('FO2 bridge: ' + result['error'])
            self.last_diagnostics = result
            return result
        except BaseException:
            # Discard the stream after a failure; a delayed response must not be
            # consumed as the next hand's decision.
            self.close()
            raise

    def action_probabilities(self, state, player_id=None):
        if player_id is not None and player_id != state.current_player():
            raise ValueError('Only the acting player may request a decision')
        game = state.get_game()
        if game is not self.game:
            validate_hulhe_game(game)
            self.game = game
        response = self.request(observation_request(state))
        values = list(response['probabilities'])
        # This is an explicit, labelled policy adaptation, never the default.
        # Preserve the raw upstream vector in the diagnostic record.
        if self.free_fold == 'check' and 0 not in state.legal_actions() and values[0] > 0:
            values[1] += values[0]
            values[0] = 0
            response['legalization'] = 'explicit_free_fold_to_check'
        try:
            return validate_probabilities(values, state.legal_actions())
        except ValueError as exc:
            raise ValueError(f'{exc}; lookup={response}') from exc

    def close(self):
        process = self._process
        if process is not None:
            self._process = None
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
        self._selector.close()
        self._errors.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
