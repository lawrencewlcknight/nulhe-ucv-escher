"""Dependency-free, predeclared inputs and budgets for frozen-target checks."""
from .p104_protocol import SOURCE_URI, POLICY_SHA256, ITERATION, TRAINER_FINGERPRINT

PROTOCOL = 'sequential_frozen_checks_v1'
CAPTURE_URI = 'gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-p104-regret-20261122-073047/capture'
MANIFEST_SHA256 = 'a347525be263a3d265286b1eba3f6a03bbedbe1fbef2d4f61b0de896b37280ab'
CAPTURE_HASHES = {
    'regret_0_before.npz': '4409b5c1d2393e95caa8de7cbe2d767f05e55b9e0355d95c1bbb64a4ac391b68',
    'regret_0_after.npz': '5aefcc9d5ae2a066aec95a6dc0d925599a4742fc4f1db3e09026d4dc34e89b43',
    'regret_1_before.npz': '4969d33eeb3bb09ed4bcf21f8d4bf7c8577a3989a201e75b41dcb67cd3ac4286',
    'regret_1_after.npz': '689ee002f5f5fc8ef5819be2bb7e0dccf019dc2a518532961cca40d26151e85a',
}
SEEDS = (240701, 240702)
ARMS = {'unrestricted': 0., 'selective': 0., 'protected_0.1': .1, 'protected_0.3': .3}
WARMUP_UPDATES = 200
BUDGETS = (0, 50, 150, 300)
POLICY_GROUPS_PER_STREET = 4096
LEARNING_RATE = .0003
BATCH_SIZE = 512
AUX_BATCH_SIZE = 128
TEACHER_REFRESH = 50
