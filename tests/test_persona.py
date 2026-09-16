"""Viora — Persona System Tests

Tests that verify the AI persona system produces measurably different
character voices for each persona.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from persona import (
    PRESET_PERSONAS,
    get_available_personas,
    get_persona,
    get_active_persona,
    set_active_persona,
    delete_user_persona,
)
from prompt_tree import build_chat_system_prompt, build_chat_user_prompt


class TestPersonaDefinitions:
    """Verify persona definitions are complete and distinct."""

    def test_all_personas_have_required_fields(self):
        """Every persona has all required fields."""
        required = [
            "id", "name", "icon", "description",
            "system_prompt", "chat_prompt_extra",
            "tone", "focus", "interaction_frequency", "caring_level",
            "empathy_level", "analysis_level", "advice_level", "humor_level",
            "proactive_strength", "burst_probability", "emoji_density",
            "forbidden", "phrase_preferences",
        ]
        for pid, p in PRESET_PERSONAS.items():
            for field in required:
                assert field in p, f"Persona '{pid}' missing field: {field}"

    def test_all_personas_have_first_message(self):
        """Every persona has a first_message_template."""
        for pid, p in PRESET_PERSONAS.items():
            assert "first_message_template" in p, f"Persona '{pid}' missing first_message_template"
            assert len(p["first_message_template"]) > 10, f"Persona '{pid}' first_message too short"

    def test_chat_prompt_extras_are_distinct(self):
        """No two personas share the same chat_prompt_extra."""
        extras = [p["chat_prompt_extra"] for p in PRESET_PERSONAS.values()]
        for i in range(len(extras)):
            for j in range(i + 1, len(extras)):
                assert extras[i] != extras[j], (
                    f"Personas {list(PRESET_PERSONAS.keys())[i]} and "
                    f"{list(PRESET_PERSONAS.keys())[j]} have identical chat_prompt_extra"
                )

    def test_system_prompts_are_distinct(self):
        """No two personas share the same system_prompt."""
        prompts = [p["system_prompt"] for p in PRESET_PERSONAS.values()]
        for i in range(len(prompts)):
            for j in range(i + 1, len(prompts)):
                assert prompts[i] != prompts[j]

    def test_tones_are_distinct(self):
        """Each persona has a unique tone."""
        tones = [p["tone"] for p in PRESET_PERSONAS.values()]
        assert len(tones) == len(set(tones))

    def test_five_personas_available(self):
        """Exactly 5 preset personas are defined."""
        assert len(PRESET_PERSONAS) == 5

    def test_persona_levels_in_range(self):
        """Empathy/analysis/advice/humor levels are 1-5."""
        level_fields = ["empathy_level", "analysis_level", "advice_level", "humor_level"]
        for pid, p in PRESET_PERSONAS.items():
            for field in level_fields:
                assert 1 <= p[field] <= 5, f"Persona '{pid}' {field}={p[field]} out of range"


class TestPersonaCharacteristics:
    """Verify persona differentiation by character traits."""

    def test_warm_friend_has_highest_empathy(self):
        """默认温暖好友共情最高."""
        empathy = {p["name"]: p["empathy_level"] for p in PRESET_PERSONAS.values()}
        assert empathy["温暖好友"] == max(empathy.values())

    def test_fitness_coach_has_high_advice(self):
        """健身教练建议倾向高."""
        coach = PRESET_PERSONAS["fitness_coach"]
        assert coach["advice_level"] >= 4

    def test_sarcastic_bff_has_highest_humor(self):
        """毒舌闺蜜幽默最高."""
        humor = {p["name"]: p["humor_level"] for p in PRESET_PERSONAS.values()}
        assert humor["毒舌闺蜜"] == max(humor.values())

    def test_analyst_has_lowest_empathy(self):
        """理性分析师共情最低（用数据说话）."""
        empathy = {p["name"]: p["empathy_level"] for p in PRESET_PERSONAS.values()}
        assert empathy["理性分析师"] == min(empathy.values())

    def test_tcm_aunt_has_lowest_humor(self):
        """中医阿姨幽默最低（温柔严肃）."""
        humor = {p["name"]: p["humor_level"] for p in PRESET_PERSONAS.values()}
        assert humor["中医阿姨"] == min(humor.values())

    def test_sarcastic_bff_highest_burst_probability(self):
        """毒舌闺蜜连发消息概率最高."""
        burst = {p["name"]: p["burst_probability"] for p in PRESET_PERSONAS.values()}
        assert burst["毒舌闺蜜"] == max(burst.values())

    def test_analyst_lowest_burst_probability(self):
        """理性分析师连发消息概率最低."""
        burst = {p["name"]: p["burst_probability"] for p in PRESET_PERSONAS.values()}
        assert burst["理性分析师"] == min(burst.values())


class TestPersonaAPI:
    """Test persona management functions."""

    def test_get_available_personas(self):
        """get_available_personas returns all 5 with minimal info."""
        personas = get_available_personas()
        assert len(personas) == 5
        for p in personas:
            assert "id" in p
            assert "name" in p
            assert "icon" in p
            assert "description" in p
            assert "tone" in p
            # Should NOT leak system_prompt
            assert "system_prompt" not in p
            assert "chat_prompt_extra" not in p

    def test_get_persona_valid(self):
        """get_persona returns full persona dict."""
        p = get_persona("fitness_coach")
        assert p["id"] == "fitness_coach"
        assert p["name"] == "健身教练"
        assert "system_prompt" in p

    def test_get_persona_fallback(self):
        """get_persona with invalid ID falls back to default."""
        p = get_persona("nonexistent")
        assert p["id"] == "default"

    def test_set_and_get_active_persona(self, tmp_path, monkeypatch):
        """set_active_persona persists and get returns correct persona."""
        import persona as pmod
        monkeypatch.setattr(pmod, "PERSONA_FILE", str(tmp_path / "persona.json"))
        monkeypatch.setattr(pmod, "_persona_file_for_user", lambda uid: str(tmp_path / f"persona_{uid}.json"))

        user_id = "test_user_123"
        result = set_active_persona("fitness_coach", user_id=user_id)
        assert result["id"] == "fitness_coach"

        active = get_active_persona(user_id=user_id)
        assert active["id"] == "fitness_coach"

    def test_delete_user_persona(self, tmp_path, monkeypatch):
        """delete_user_persona removes saved preference."""
        import persona as pmod
        monkeypatch.setattr(pmod, "PERSONA_FILE", str(tmp_path / "persona.json"))
        monkeypatch.setattr(pmod, "_persona_file_for_user", lambda uid: str(tmp_path / f"persona_{uid}.json"))

        user_id = "test_user_del"
        set_active_persona("sarcastic_bff", user_id=user_id)
        delete_user_persona(user_id=user_id)
        active = get_active_persona(user_id=user_id)
        assert active["id"] == "default"

    def test_set_invalid_persona_falls_back(self, tmp_path, monkeypatch):
        """Setting invalid persona ID falls back to default."""
        import persona as pmod
        monkeypatch.setattr(pmod, "PERSONA_FILE", str(tmp_path / "persona.json"))
        monkeypatch.setattr(pmod, "_persona_file_for_user", lambda uid: str(tmp_path / f"persona_{uid}.json"))

        result = set_active_persona("invalid_persona", user_id="test_user_inv")
        assert result["id"] == "default"


class TestPersonaConfigRobustness:
    """Corrupt/legacy persona.json must degrade gracefully, not 500.

    persona.json is hand-editable and may be partially written. Pre-fix:
    - a valid-JSON non-object top level (list/str/int/null) made
      _load_user_persona call ``data.get`` → AttributeError (uncaught);
    - a non-string persona_id (list/dict) was returned as-is and reached
      PRESET_PERSONAS.get() → TypeError (unhashable), crashing
      get_active_persona instead of falling back to the default persona.
    """

    def _point_at(self, monkeypatch, tmp_path, content: str):
        import persona as pmod
        path = tmp_path / "persona_robust.json"
        path.write_text(content, encoding="utf-8")
        monkeypatch.setattr(pmod, "_persona_file_for_user", lambda uid: str(path))
        return pmod

    @pytest.mark.parametrize("content", ["[]", '["default"]', '"default"', "42", "null"])
    def test_non_object_top_level_falls_back(self, tmp_path, monkeypatch, content):
        """Top-level non-dict JSON yields default persona, no AttributeError."""
        pmod = self._point_at(monkeypatch, tmp_path, content)
        assert pmod._load_user_persona("u") is None
        assert get_active_persona(user_id="u")["id"] == "default"

    @pytest.mark.parametrize("content", ['{"persona_id": ["default"]}',
                                         '{"persona_id": {"x": 1}}',
                                         '{"persona_id": 123}',
                                         '{"persona_id": null}'])
    def test_non_string_persona_id_falls_back(self, tmp_path, monkeypatch, content):
        """Non-string persona_id must not reach PRESET_PERSONAS.get()."""
        pmod = self._point_at(monkeypatch, tmp_path, content)
        assert pmod._load_user_persona("u") is None
        assert get_active_persona(user_id="u")["id"] == "default"

    def test_missing_persona_id_key_falls_back(self, tmp_path, monkeypatch):
        pmod = self._point_at(monkeypatch, tmp_path, '{"other": "x"}')
        assert pmod._load_user_persona("u") is None

    def test_well_formed_persona_id_unchanged(self, tmp_path, monkeypatch):
        """Well-formed stored id is returned byte-identically and honored."""
        pmod = self._point_at(monkeypatch, tmp_path, '{"persona_id": "fitness_coach"}')
        assert pmod._load_user_persona("u") == "fitness_coach"
        assert get_active_persona(user_id="u")["id"] == "fitness_coach"

    def test_corrupt_json_falls_back(self, tmp_path, monkeypatch):
        pmod = self._point_at(monkeypatch, tmp_path, "{not json")
        assert pmod._load_user_persona("u") is None
        assert get_active_persona(user_id="u")["id"] == "default"


class TestPersonaPromptBuilding:
    """Test system and user prompt construction."""

    def test_build_chat_system_prompt_includes_persona(self):
        """System prompt includes persona's system_prompt."""
        pid = "default"
        prompt = build_chat_system_prompt(pid, time_context="当前时间: 2026-01-01 10:00")
        assert "安全边界" in prompt
        assert "2026-01-01 10:00" in prompt

    def test_build_chat_system_prompt_no_time_context(self):
        """System prompt works without time context."""
        prompt = build_chat_system_prompt("analyst")
        assert "安全边界" in prompt

    def test_build_chat_user_prompt_includes_persona_extra(self):
        """User prompt includes persona's chat_prompt_extra."""
        prompt = build_chat_user_prompt(
            "fitness_coach",
            message="今天没运动",
            extracted={"energy": {"score": 3, "reason": "有点累"}},
            context=[],
        )
        assert "今天没运动" in prompt

    def test_different_personas_produce_different_prompts(self):
        """Different personas produce observably different system prompts."""
        prompt_default = build_chat_system_prompt("default")
        prompt_coach = build_chat_system_prompt("fitness_coach")
        prompt_bff = build_chat_system_prompt("sarcastic_bff")
        prompt_analyst = build_chat_system_prompt("analyst")
        prompt_tcm = build_chat_system_prompt("tcm_aunt")

        all_prompts = [prompt_default, prompt_coach, prompt_bff, prompt_analyst, prompt_tcm]
        unique = set(all_prompts)
        assert len(unique) == 5, "All 5 personas should produce unique system prompts"

    def test_different_personas_produce_different_user_prompts(self):
        """Different personas produce different user prompt instructions."""
        base_args = {
            "message": "今天状态一般",
            "extracted": {"energy": {"score": 4, "reason": "普通"}},
            "context": [],
        }
        prompts = [
            build_chat_user_prompt(pid, **base_args)
            for pid in ["default", "fitness_coach", "tcm_aunt", "sarcastic_bff", "analyst"]
        ]
        unique = set(prompts)
        assert len(unique) == 5, "All 5 personas should produce unique user prompts"
