import os
import tempfile

# must run before anything imports aml_copilot.config
_tmp = tempfile.mkdtemp(prefix="aml_test_")
os.environ["DATA_DIR"] = os.path.join(_tmp, "data")
os.environ["MODEL_DIR"] = os.path.join(_tmp, "models")

import pytest  # noqa: E402

from aml_copilot.transactions import db, synth, train  # noqa: E402


@pytest.fixture(scope="session")
def trained():
    accounts, txns = synth.generate(n_accounts=1500, seed=3)
    db.load_frames(accounts, txns)
    return train.train()
