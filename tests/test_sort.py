import pytest

from magpie.config import load_config
from magpie.sort import classify
from tests.conftest import FakeModel


def categories(folder):
    return {r.path.stem: r.category for r in classify(folder, load_config(folder), FakeModel())}


def test_classify_assigns_matching_category_and_other(folder):
    assert categories(folder) == {"red": "red", "blue": "blue", "green": "other"}


def test_classify_reuses_cached_embeddings(folder, monkeypatch):
    categories(folder)
    monkeypatch.setattr(FakeModel, "embed_images", lambda *_: pytest.fail("re-embedded"))
    assert categories(folder)["red"] == "red"


def test_config_rejects_reserved_name(folder):
    (folder / "magpie.toml").write_text('[[category]]\nname = "other"\ndescription = "x"\n')
    with pytest.raises(ValueError, match="reserved"):
        load_config(folder)
