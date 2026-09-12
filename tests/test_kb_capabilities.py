"""Stage B/C — capability tags: which permission a unit needs, and who may see it."""
from app.kb_capabilities import EVERYONE, capability_from_requires, visible


def test_everyone_is_visible_to_all_even_without_capabilities():
    assert visible(EVERYONE, frozenset())
    assert visible(EVERYONE, frozenset({"company.admin"}))


def test_restricted_unit_needs_the_capability():
    assert visible("company.admin", frozenset({"company.admin", "everyone"}))
    assert not visible("company.admin", frozenset({"everyone"}))
    assert not visible("company.admin", frozenset())


def test_capability_derived_from_the_requires_line():
    assert capability_from_requires("a Company administrator") == "company.admin"
    assert capability_from_requires(
        "a Checklist editor role (Company administrator, Corporate manager, ...)") == "checklists.edit"
    assert capability_from_requires(None) == EVERYONE
    assert capability_from_requires("") == EVERYONE


def test_explicit_capability_wins_over_requires():
    assert capability_from_requires("a Company administrator", explicit="tasks.create") == "tasks.create"
