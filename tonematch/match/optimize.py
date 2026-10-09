"""Mixed categorical/continuous search over plugin parameters with Optuna (ask/tell).

Every take in Live is real time, so several tracks are optimized in lock-step: each
round asks every track's study for a candidate, applies them all, records all
tracks in one pass, then tells each study its own loss.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import optuna

from ..config import ParamSpec
from ..live.params import LiveParam, resolve, resolve_group

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class Knob:
    """One optimizable parameter on one device."""
    track: str                  # Live track the device sits on
    device: str                 # device key on that track (e.g. "archetype_gojira")
    spec: ParamSpec
    params: list[LiveParam]     # all params of that device, for {placeholder} resolution
    db_range: tuple[float, float] | None = None   # set: value is in dB, applied via display

    @property
    def uid(self) -> str:
        return f"{self.track}/{self.device}"


@dataclass
class Setting:
    track: str
    device: str
    param_index: int
    value: float                # native value, or dB when is_db
    label: str
    is_db: bool = False


@dataclass
class Candidate:
    trial: optuna.trial.Trial
    values: list[Setting] = field(default_factory=list)


class ToneSearch:
    def __init__(self, name: str, knobs: list[Knob], seed: int = 0, n_startup: int = 8):
        self.name = name
        self.knobs = knobs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup, multivariate=True,
                                                 group=True)
        self.study = optuna.create_study(direction="minimize", sampler=sampler, study_name=name)
        self._seed_current()
        self._seed_categoricals()

    @property
    def empty(self) -> bool:
        return not self.knobs

    def _seed_current(self) -> None:
        """First trial = the settings the plug-ins have now, so the search can't end up worse
        than where it started."""
        current: dict = {}
        for knob in self.knobs:
            if knob.db_range is not None or "{" in knob.spec.pattern:
                continue
            members = (resolve_group(knob.spec, knob.params) if knob.spec.group
                       else [x for x in [resolve(knob.spec, knob.params)] if x is not None])
            if not members:
                continue
            p = members[0]
            if knob.spec.kind == "categorical":
                current[self._pname(knob, None)] = int(round(p.value))
            else:
                lo, hi = knob.spec.bounds(p.normalized(p.value))
                current[self._pname(knob, None)] = min(hi, max(lo, p.normalized(p.value)))
        if current:
            self.study.enqueue_trial(current)

    def _seed_categoricals(self) -> None:
        """Make sure every option of the first categorical (e.g. the amp) is tried once
        early, with the remaining knobs at their current values."""
        cats = [k for k in self.knobs if k.spec.kind == "categorical"]
        if not cats:
            return
        first = cats[0]
        p = resolve(first.spec, first.params)
        if p is None:
            return
        for native, _label in p.choices():
            self.study.enqueue_trial({self._pname(first, None): native}, skip_if_exists=True)

    @staticmethod
    def _pname(knob: Knob, label: str | None) -> str:
        base = f"{knob.uid}.{knob.spec.key}"
        return f"{base}@{label}" if label else base

    def ask(self) -> Candidate:
        trial = self.study.ask()
        cand = Candidate(trial)
        choices: dict[str, dict[str, str]] = {}   # device uid -> {categorical key: label}
        for knob in sorted(self.knobs, key=lambda k: k.spec.kind != "categorical"):
            dev_choices = choices.setdefault(knob.uid, {})
            members = (resolve_group(knob.spec, knob.params, dev_choices) if knob.spec.group
                       else [x for x in [resolve(knob.spec, knob.params, dev_choices)] if x is not None])
            if not members:
                continue
            p = members[0]
            if knob.spec.kind == "categorical":
                options = p.choices()
                native = trial.suggest_categorical(self._pname(knob, None), [v for v, _ in options])
                label = dict(options)[native]
                dev_choices[knob.spec.key] = label
                cand.values.append(Setting(knob.track, knob.device, p.index, float(native), label))
            elif knob.db_range is not None:
                db = trial.suggest_float(self._pname(knob, None), *knob.db_range)
                cand.values.append(Setting(knob.track, knob.device, p.index, db, f"{p.name}={db:+.1f} dB",
                                           is_db=True))
            else:
                # Conditional knobs (pattern uses {placeholders}) get one distribution per choice.
                dep = "+".join(dev_choices[k] for k in sorted(dev_choices) if "{" + k + "}" in knob.spec.pattern)
                # `span` knobs search around the value the parameter had when the knob was made.
                lo, hi = knob.spec.bounds(p.normalized(p.value))
                u = trial.suggest_float(self._pname(knob, dep or None), lo, hi)
                for m in members:   # a group knob moves all its parameters together
                    cand.values.append(Setting(knob.track, knob.device, m.index, m.native(u), f"{m.name}={u:.3f}"))
        return cand

    def tell(self, cand: Candidate, loss: float) -> None:
        self.study.tell(cand.trial, loss)

    def best(self) -> tuple[float, dict] | None:
        try:
            t = self.study.best_trial
        except ValueError:
            return None
        return t.value, t.params

    def replay(self, params: dict) -> Candidate:
        """Rebuild the parameter assignment of a finished trial (to apply the best one).
        The replay trial is closed as pruned so it never counts as a result."""
        self.study.enqueue_trial(params)
        cand = self.ask()
        self.study.tell(cand.trial, state=optuna.trial.TrialState.PRUNED)
        return cand


def run_lockstep(searches: list[ToneSearch], n_trials: int,
                 evaluate: Callable[[dict[str, Candidate]], dict[str, float]],
                 on_round: Callable[[int, dict[str, float]], None] | None = None) -> None:
    """Drive several studies together; `evaluate` applies all candidates, renders once,
    and returns a loss per search name."""
    active = [s for s in searches if not s.empty]
    for i in range(n_trials):
        cands = {s.name: s.ask() for s in active}
        losses = evaluate(cands)
        for s in active:
            s.tell(cands[s.name], losses[s.name])
        if on_round:
            on_round(i, losses)
