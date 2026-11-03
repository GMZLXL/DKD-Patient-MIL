"""Post-hoc sensitivity protocols; historical split and selection code is untouched."""
from dataclasses import dataclass, field
import math
from sklearn.model_selection import train_test_split

METHODS = {
    'acmil_official_ratio': 'ACMIL-GA-adapted-v2-official-mask',
    'dgr_fullobj': 'DGR-MIL-adapted-fullobj-v2',
}


def inner_split_without_test_labels(bags, fold, seed=42):
    """Same stratified 8:2 development split; never inspect held-out labels/features."""
    dev = sorted(pid for pid, bag in bags.items() if bag['fold'] != fold)
    test = sorted(set(bags) - set(dev))
    if not dev or not test:
        raise ValueError('Unknown or empty outer fold')
    train, val = train_test_split(dev, test_size=.2, random_state=seed,
                                 stratify=[bags[pid]['y'] for pid in dev])
    result = dict(train=sorted(train), val=sorted(val), test=test)
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise RuntimeError('Patient leakage')
    for ids in [train, val]:
        if {bags[pid]['y'] for pid in ids} != {0, 1}:
            raise ValueError('Development subsets require both classes')
    return result


@dataclass
class CheckpointPolicy:
    auxiliary_start_epoch: int = 10
    checkpoint_eligible_epoch: int = 10
    patience: int = 20
    require_auxiliary: bool = True
    best_epoch: int | None = field(default=None, init=False)
    best_score: tuple = field(default=(-math.inf, -math.inf), init=False)
    stale: int = field(default=0, init=False)
    last_epoch: int = field(default=0, init=False)

    def __post_init__(self):
        if self.patience < 1 or self.checkpoint_eligible_epoch < 1:
            raise ValueError('Invalid eligibility/patience')
        if self.require_auxiliary and self.checkpoint_eligible_epoch < self.auxiliary_start_epoch:
            raise ValueError('Checkpoint eligibility cannot precede the auxiliary objective')

    def observe(self, epoch, val_auc, val_ce, *, auxiliary_active=False,
                auxiliary_optimizer_steps=0):
        if epoch <= self.last_epoch:
            raise ValueError('Epochs must increase')
        if not math.isfinite(val_auc) or not math.isfinite(val_ce):
            raise ValueError('Validation score is nonfinite')
        self.last_epoch = epoch
        eligible = epoch >= self.checkpoint_eligible_epoch
        if eligible and self.require_auxiliary and not (
                epoch >= self.auxiliary_start_epoch and auxiliary_active and auxiliary_optimizer_steps > 0):
            raise RuntimeError('Eligible DGR checkpoint requires a completed auxiliary-objective update')
        save = False
        if eligible:
            score = (float(val_auc), -float(val_ce))
            if score > self.best_score:
                self.best_score, self.best_epoch, self.stale = score, epoch, 0
                save = True
            else:
                self.stale += 1
        return dict(epoch=epoch, checkpoint_eligible=eligible, save_best=save,
                    eligible_stale=self.stale, stop=eligible and self.stale >= self.patience,
                    best_epoch=self.best_epoch,
                    auxiliary_start_epoch=self.auxiliary_start_epoch,
                    checkpoint_eligible_epoch=self.checkpoint_eligible_epoch,
                    auxiliary_active=bool(auxiliary_active),
                    auxiliary_optimizer_steps=auxiliary_optimizer_steps)

    def validate_checkpoint(self, checkpoint):
        if self.best_epoch is None:
            raise RuntimeError('No eligible checkpoint; warmup-only runs are incomplete')
        if checkpoint['epoch'] != self.best_epoch or checkpoint['epoch'] < self.checkpoint_eligible_epoch:
            raise RuntimeError('Ineligible or mismatched best checkpoint')
        if self.require_auxiliary and not (
                checkpoint['auxiliary_active_at_best_checkpoint']
                and checkpoint['auxiliary_optimizer_steps'] > 0):
            raise RuntimeError('Selected checkpoint has no auxiliary training evidence')


def policy_for(method, patience=20):
    if method == 'dgr_fullobj':
        return CheckpointPolicy(patience=patience)
    if method == 'acmil_official_ratio':
        return CheckpointPolicy(auxiliary_start_epoch=0, checkpoint_eligible_epoch=1,
                                patience=patience, require_auxiliary=False)
    raise ValueError('Unknown versioned method')
