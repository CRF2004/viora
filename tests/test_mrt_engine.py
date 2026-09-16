"""
test_mrt_engine.py — MRT validation framework tests (Phase D).
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from mrt_engine import (
    MRTrialManager,
    MRTDecision,
    INTERVENTION_CATALOG,
)


class TestMRTDecision:
    """MRTDecision data model tests."""

    def test_default_creation(self):
        d = MRTDecision(
            timestamp="2026-07-13T10:00:00+08:00",
            user_id="u1",
            intervenable_state="willing_but_stuck",
            chosen_intervention="lower_barrier",
            available_interventions=["lower_barrier", "gentle_suggest"],
        )
        assert d.user_id == "u1"
        assert d.chosen_intervention == "lower_barrier"
        assert d.proximal_responded is None

    def test_to_dict_roundtrip(self):
        d = MRTDecision(
            timestamp="2026-07-13T10:00:00+08:00",
            user_id="u1",
            intervenable_state="normal",
            chosen_intervention="gentle_suggest",
            available_interventions=["gentle_suggest", "lower_barrier"],
            proximal_responded=True,
            proximal_willingness_delta=0.5,
        )
        data = d.to_dict()
        restored = MRTDecision.from_dict(data)
        assert restored.user_id == "u1"
        assert restored.chosen_intervention == "gentle_suggest"
        assert restored.proximal_responded is True
        assert restored.proximal_willingness_delta == 0.5


class TestMRTrialManager:
    """MRTrialManager CRUD and randomization tests."""

    def test_fresh_manager(self):
        m = MRTrialManager("u1")
        assert m.user_id == "u1"
        assert len(m.decisions) == 0

    def test_randomize_returns_valid(self):
        m = MRTrialManager("u1")
        chosen = m.randomize("low_mood_available",
                              available=["gentle_suggest", "lower_barrier"])
        assert chosen in ("gentle_suggest", "lower_barrier")
        assert len(m.decisions) == 1
        assert m.decisions[0].intervenable_state == "low_mood_available"

    def test_randomize_all_interventions(self):
        m = MRTrialManager("u1")
        chosen = m.randomize("normal")
        assert chosen in INTERVENTION_CATALOG

    def test_record_outcome(self):
        m = MRTrialManager("u1")
        chosen = m.randomize("normal", available=["gentle_suggest"])
        assert chosen == "gentle_suggest"

        m.record_outcome("gentle_suggest", responded=True,
                         willingness_delta=2.0)
        d = m.decisions[0]
        assert d.proximal_responded is True
        assert d.proximal_willingness_delta == 2.0
        assert d.proximal_outcome_ts is not None

    def test_record_outcome_latest_only(self):
        """record_outcome matches the most recent unmatched decision."""
        m = MRTrialManager("u1")
        m.randomize("normal", available=["gentle_suggest"])
        m.randomize("normal", available=["gentle_suggest"])
        m.record_outcome("gentle_suggest", responded=True)
        # Only the latest should be matched
        assert m.decisions[0].proximal_responded is None
        assert m.decisions[1].proximal_responded is True

    def test_estimate_effect_no_data(self):
        m = MRTrialManager("u1")
        effect = m.estimate_effect("gentle_suggest")
        assert effect["n_trials"] == 0
        assert "no_data" in effect.get("error", "")

    def test_estimate_effect_with_data(self):
        m = MRTrialManager("u1")
        # Force 3 gentle_suggest decisions
        for _ in range(3):
            m.randomize("normal", available=["gentle_suggest"])
        for d in m.decisions:
            d.proximal_responded = True
            d.proximal_willingness_delta = 1.0
            d.proximal_outcome_ts = "2026-07-13T12:00:00+08:00"

        effect = m.estimate_effect("gentle_suggest")
        assert effect["n_trials"] == 3
        assert effect["mean_response_rate"] == 1.0
        assert effect["mean_willingness_delta"] == 1.0

    def test_n_of_1_analysis(self):
        m = MRTrialManager("u1")
        # Create enough trials with deterministic intervention assignment
        for iv in ["lower_barrier", "gentle_suggest"]:
            for _ in range(3):
                chosen = m.randomize("normal", available=[iv])
                m.record_outcome(chosen, responded=True, willingness_delta=1.0)

        result = m.n_of_1_analysis(min_trials=2)
        assert result["total_decisions"] == 6
        assert len(result["rankings"]) > 0
        assert result["best_intervention"] is not None

    def test_save_load_roundtrip(self, tmp_path):
        """Save and reload preserves state."""
        import os
        from mrt_engine import MRT_DATA_FILE

        original_file = MRT_DATA_FILE
        test_file = str(tmp_path / "mrt_test.json")

        # Monkey-patch the file path
        import mrt_engine as mrt_mod
        mrt_mod.MRT_DATA_FILE = test_file

        m = mrt_mod.MRTrialManager("u1")
        m.randomize("normal", available=["gentle_suggest"])
        m.record_outcome("gentle_suggest", responded=True)
        m.save()

        m2 = mrt_mod.MRTrialManager.load("u1")
        assert len(m2.decisions) == 1
        assert m2.decisions[0].chosen_intervention == "gentle_suggest"
        assert m2.decisions[0].proximal_responded is True

        # Restore
        mrt_mod.MRT_DATA_FILE = original_file


class TestInterventionCatalog:
    """Verify the intervention catalog completeness."""

    def test_all_interventions_have_entries(self):
        """Every intervention in the catalog has target_dim and hypothesis."""
        for name, info in INTERVENTION_CATALOG.items():
            assert "target_dim" in info, f"{name} missing target_dim"
            assert "hypothesis" in info, f"{name} missing hypothesis"
            assert "expected_effect" in info, f"{name} missing expected_effect"
            assert info["expected_effect"] in ("positive", "negative")


class TestMRTCorruptTolerance:
    """Legacy/corrupt mrt_data.json must not crash load()/save()."""

    @pytest.fixture
    def isolated_mrt(self, tmp_path, monkeypatch):
        import mrt_engine as mrt_mod
        monkeypatch.setattr(mrt_mod, "MRT_DATA_FILE", str(tmp_path / "mrt_data.json"))
        return tmp_path

    def _write(self, tmp_path, payload):
        (tmp_path / "mrt_data.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def test_load_non_dict_store_top_level(self, isolated_mrt):
        import mrt_engine as mrt_mod
        self._write(isolated_mrt, ["junk"])
        m = mrt_mod.MRTrialManager.load("u1")
        assert m.decisions == []
        # save() must repair the store into a dict without raising.
        m.save()
        stored = json.loads((isolated_mrt / "mrt_data.json").read_text("utf-8"))
        assert isinstance(stored, dict) and "u1" in stored

    def test_load_non_dict_user_record(self, isolated_mrt):
        import mrt_engine as mrt_mod
        self._write(isolated_mrt, {"u1": "junk"})
        assert mrt_mod.MRTrialManager.load("u1").decisions == []

    def test_load_non_list_decisions(self, isolated_mrt):
        import mrt_engine as mrt_mod
        self._write(isolated_mrt, {"u1": {"decisions": "junk"}})
        assert mrt_mod.MRTrialManager.load("u1").decisions == []

    def test_load_skips_malformed_decisions_keeps_valid(self, isolated_mrt):
        import mrt_engine as mrt_mod
        valid = MRTDecision(
            timestamp="2026-07-13T10:00:00+08:00",
            user_id="u1",
            intervenable_state="normal",
            chosen_intervention="gentle_suggest",
            available_interventions=["gentle_suggest"],
        ).to_dict()
        self._write(isolated_mrt, {"u1": {"decisions": [
            "junk", 42, None, {"unrelated": 1}, valid,
        ]}})
        m = mrt_mod.MRTrialManager.load("u1")
        assert len(m.decisions) == 1
        assert m.decisions[0].chosen_intervention == "gentle_suggest"
