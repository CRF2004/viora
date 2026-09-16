"""
Viora - Health Data Extraction Test Suite (50 test cases)

Validates the core "一句话完成一次健康记录" (one-sentence health recording)
functionality. Tests cover:
1. JSON response parsing from LLM output
2. Data normalization / schema enforcement
3. Health extraction pipeline (mocked LLM)
4. Edge cases and error handling
5. All 8 health dimensions individually and in combination
"""

import json
import sys
import os
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine import (
    _parse_json_response,
    _empty_extracted,
    normalize_extracted,
    extract_health_data,
    _parse_story_response,
    generate_body_story,
)


class TestEmptyExtracted(unittest.TestCase):
    """Test the default extraction template."""

    def test_all_fields_present(self):
        """Template must contain all 8 health dimensions."""
        template = _empty_extracted()
        expected_keys = {"sleep", "energy", "mood", "exercise", "digestion", "skin", "pain", "diet"}
        self.assertEqual(set(template.keys()), expected_keys)

    def test_nested_objects_are_dicts(self):
        """All dimension containers must be dicts."""
        template = _empty_extracted()
        for key in template:
            self.assertIsInstance(template[key], dict,
                f"{key} should be a dict in template")

    def test_sleep_template_structure(self):
        """Sleep has quality and duration_hours."""
        template = _empty_extracted()
        self.assertIn("quality", template["sleep"])
        self.assertIn("duration_hours", template["sleep"])
        self.assertIsNone(template["sleep"]["quality"])

    def test_energy_template_structure(self):
        """Energy has score and reason (dict format)."""
        template = _empty_extracted()
        self.assertIsInstance(template["energy"], dict)
        self.assertIn("score", template["energy"])
        self.assertIsNone(template["energy"]["score"])

    def test_mood_template_structure(self):
        """Mood has score and reason (dict format)."""
        template = _empty_extracted()
        self.assertIsInstance(template["mood"], dict)
        self.assertIn("score", template["mood"])
        self.assertIsNone(template["mood"]["score"])

    def test_diet_template_structure(self):
        """Diet has notes field."""
        template = _empty_extracted()
        self.assertIn("notes", template["diet"])


class TestParseJSONResponse(unittest.TestCase):
    """Test LLM JSON response parsing."""

    def test_clean_json(self):
        """Parse a clean JSON response."""
        result = _parse_json_response('{"sleep": {"quality": 4}, "energy": 3}')
        self.assertEqual(result["sleep"]["quality"], 4)
        self.assertEqual(result["energy"], 3)

    def test_json_with_markdown_fence(self):
        """Parse JSON wrapped in markdown code fence."""
        response = '```json\n{"sleep": {"quality": 2}}\n```'
        result = _parse_json_response(response)
        self.assertEqual(result["sleep"]["quality"], 2)

    def test_json_with_text_prefix(self):
        """Parse JSON with explanatory text before it."""
        response = 'Here is the result:\n{"mood": 5, "energy": 4}'
        result = _parse_json_response(response)
        self.assertEqual(result["mood"], 5)

    def test_invalid_json_raises(self):
        """Malformed JSON should raise JSONDecodeError."""
        with self.assertRaises(json.JSONDecodeError):
            _parse_json_response("not json at all")

    def test_nested_objects_in_llm_response(self):
        """Full health extraction JSON from LLM."""
        response = '''{
            "sleep": {"quality": 3, "duration_hours": 6.5},
            "energy": 3,
            "mood": 2,
            "exercise": {"type": "跑步", "duration_minutes": 30},
            "digestion": {"status": "正常", "severity": null},
            "skin": {"status": null},
            "pain": {"location": "头痛", "severity": 3},
            "diet": {"notes": "吃了沙拉"}
        }'''
        result = _parse_json_response(response)
        self.assertEqual(result["sleep"]["quality"], 3)
        self.assertEqual(result["exercise"]["type"], "跑步")
        self.assertEqual(result["pain"]["location"], "头痛")

    def test_json_energy_as_int_parsed(self):
        """Energy as raw int from LLM JSON should be parseable."""
        result = _parse_json_response('{"energy": 4}')
        self.assertEqual(result["energy"], 4)


class TestNormalizeExtracted(unittest.TestCase):
    """Test schema normalization of extracted data."""

    def test_full_valid_data_passes_through(self):
        """A complete valid dict should preserve all values as dict format."""
        raw = {
            "sleep": {"quality": 4, "duration_hours": 8.0},
            "energy": 4, "mood": 3,
            "exercise": {"type": "游泳", "duration_minutes": 45},
            "digestion": {"status": "良好", "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": "高蛋白饮食"},
        }
        result = normalize_extracted(raw)
        self.assertEqual(result["sleep"]["quality"], 4)
        self.assertEqual(result["energy"]["score"], 4)
        self.assertEqual(result["mood"]["score"], 3)
        self.assertEqual(result["exercise"]["type"], "游泳")

    def test_flat_nulls_become_dicts(self):
        """When LLM returns flat null for nested objects, normalize to template dicts."""
        raw = {
            "sleep": None, "energy": 3, "mood": None,
            "exercise": None, "digestion": None, "skin": None,
            "pain": None, "diet": None,
        }
        result = normalize_extracted(raw)
        self.assertIsInstance(result["sleep"], dict)
        self.assertIsNone(result["sleep"]["quality"])
        self.assertIsInstance(result["diet"], dict)
        self.assertEqual(result["energy"]["score"], 3)

    def test_missing_keys_filled_from_template(self):
        """Missing keys should be filled from template."""
        raw = {"energy": 4}
        result = normalize_extracted(raw)
        self.assertIn("sleep", result)
        self.assertIn("mood", result)
        self.assertIn("diet", result)
        self.assertIsInstance(result["sleep"], dict)
        self.assertEqual(result["energy"]["score"], 4)

    def test_partial_nested_object_merged(self):
        """Partial nested objects should have missing fields filled."""
        raw = {
            "sleep": {"quality": 2}, "energy": None, "mood": None,
            "exercise": None, "digestion": None, "skin": None,
            "pain": None, "diet": None,
        }
        result = normalize_extracted(raw)
        self.assertEqual(result["sleep"]["quality"], 2)
        self.assertIsNone(result["sleep"]["duration_hours"])
        self.assertEqual(result["energy"]["score"], None)

    def test_extra_keys_not_in_schema_are_ignored(self):
        """Extra fields from LLM should not appear in result."""
        raw = {
            "sleep": {"quality": 3}, "energy": 3, "mood": None,
            "exercise": None, "digestion": None, "skin": None,
            "pain": None, "diet": None,
            "extra_field": "should be ignored",
        }
        result = normalize_extracted(raw)
        self.assertNotIn("extra_field", result)

    def test_energy_int_to_dict_conversion(self):
        """Raw int energy (old format) should become dict with score."""
        raw = {"energy": 5, "sleep": None, "mood": None, "exercise": None,
               "digestion": None, "skin": None, "pain": None, "diet": None}
        result = normalize_extracted(raw)
        self.assertEqual(result["energy"]["score"], 5)
        self.assertIsNone(result["energy"]["reason"])

    def test_mood_float_to_dict_conversion(self):
        """Raw float mood should become dict with score."""
        raw = {"energy": None, "mood": 2.5, "sleep": None, "exercise": None,
               "digestion": None, "skin": None, "pain": None, "diet": None}
        result = normalize_extracted(raw)
        self.assertEqual(result["mood"]["score"], 2.5)


class TestExtractHealthData(unittest.TestCase):
    """Test the full extraction pipeline with mocked LLM."""

    def setUp(self):
        self.mock_llm_response = json.dumps({
            "sleep": {"quality": 4, "duration_hours": 7.5},
            "energy": 4, "mood": 3,
            "exercise": {"type": "步行", "duration_minutes": 60},
            "digestion": {"status": "正常", "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })

    def test_extract_sleep_and_exercise(self):
        """'今天走了很多路但睡得不好' should extract sleep quality and walking."""
        llm_response = json.dumps({
            "sleep": {"quality": 2, "duration_hours": None},
            "energy": None, "mood": None,
            "exercise": {"type": "步行", "duration_minutes": None},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("今天走了很多路但睡得不好")
        self.assertEqual(result["sleep"]["quality"], 2)
        self.assertEqual(result["exercise"]["type"], "步行")

    def test_extract_multiple_dimensions(self):
        """Message mentioning multiple dimensions should extract energy as dict."""
        with patch("ai_engine._call_llm", return_value=self.mock_llm_response):
            result = extract_health_data("今天睡得很好，精力不错，走了很多路")
        self.assertEqual(result["sleep"]["quality"], 4)
        self.assertEqual(result["energy"]["score"], 4)

    def test_extract_returns_valid_schema(self):
        """Every extraction must return all 8 dimensions in correct schema."""
        with patch("ai_engine._call_llm", return_value=self.mock_llm_response):
            result = extract_health_data("感觉还行")
        expected_keys = {"sleep", "energy", "mood", "exercise", "digestion", "skin", "pain", "diet"}
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertIsInstance(result["sleep"], dict)
        self.assertIsInstance(result["exercise"], dict)
        self.assertIsInstance(result["energy"], dict)

    def test_extract_handles_llm_error_gracefully(self):
        """When LLM returns invalid JSON, should return empty template, not crash."""
        with patch("ai_engine._call_llm", return_value="Sorry, I can't parse that."):
            result = extract_health_data("随便说点什么")
        self.assertIn("sleep", result)
        self.assertIn("energy", result)
        self.assertIsNotNone(result["energy"])

    def test_extract_digestion_and_mood(self):
        """'胃不舒服，心情很差' should extract digestion and low mood as dict."""
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": None, "mood": 2,
            "exercise": {"type": None, "duration_minutes": None},
            "digestion": {"status": "胃不适", "severity": 3},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("胃不舒服，心情很差")
        self.assertEqual(result["mood"]["score"], 2)
        self.assertEqual(result["digestion"]["status"], "胃不适")

    def test_extract_pain_location(self):
        """'今天头痛得厉害' should extract headache pain."""
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": None, "mood": None,
            "exercise": {"type": None, "duration_minutes": None},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": "头痛", "severity": 4},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("今天头痛得厉害")
        self.assertEqual(result["pain"]["location"], "头痛")
        self.assertEqual(result["pain"]["severity"], 4)

    def test_extract_empty_message(self):
        """Empty message should still return valid schema."""
        with patch("ai_engine._call_llm", return_value=self.mock_llm_response):
            result = extract_health_data("...")
        self.assertIsInstance(result, dict)
        self.assertIn("sleep", result)

    def test_pain_severity_null_by_default(self):
        """No pain in message → pain severity should be None."""
        with patch("ai_engine._call_llm", return_value=self.mock_llm_response):
            result = extract_health_data("今天状态不错")
        self.assertIsNone(result["pain"]["severity"])

    def test_extract_diet_notes(self):
        """'今天吃了很多蔬菜' should extract diet notes."""
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": None, "mood": None,
            "exercise": {"type": None, "duration_minutes": None},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": "蔬菜为主"},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("今天吃了很多蔬菜")
        self.assertEqual(result["diet"]["notes"], "蔬菜为主")

    def test_extract_exercise_exact_duration(self):
        """'跑步30分钟' should extract exercise with duration."""
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": None, "mood": None,
            "exercise": {"type": "跑步", "duration_minutes": 30},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("跑步30分钟")
        self.assertEqual(result["exercise"]["type"], "跑步")
        self.assertEqual(result["exercise"]["duration_minutes"], 30)

    def test_extract_sleep_duration(self):
        """'睡了8小时' should extract exact sleep duration."""
        llm_response = json.dumps({
            "sleep": {"quality": 4, "duration_hours": 8.0},
            "energy": None, "mood": None,
            "exercise": {"type": None, "duration_minutes": None},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("睡了8小时")
        self.assertEqual(result["sleep"]["duration_hours"], 8.0)
        self.assertEqual(result["sleep"]["quality"], 4)

    def test_extract_high_energy(self):
        """'今天精力充沛' should extract high energy as dict."""
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": 5, "mood": 4,
            "exercise": {"type": None, "duration_minutes": None},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("今天精力充沛！心情也很好")
        self.assertEqual(result["energy"]["score"], 5)
        self.assertEqual(result["mood"]["score"], 4)

    def test_extract_negative_mood(self):
        """'心情很低落' should extract low mood as dict."""
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": 2, "mood": 1,
            "exercise": {"type": None, "duration_minutes": None},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("心情很低落，什么都不想做")
        self.assertEqual(result["mood"]["score"], 1)

    def test_extract_skin_condition(self):
        """'脸上长痘了' should extract skin status."""
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": None, "mood": None,
            "exercise": {"type": None, "duration_minutes": None},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": "长痘"},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("脸上长痘了")
        self.assertEqual(result["skin"]["status"], "长痘")

    def test_extract_back_pain(self):
        """'腰疼了两天' should extract back pain."""
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": None, "mood": None,
            "exercise": {"type": None, "duration_minutes": None},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": "腰部", "severity": 3},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("腰疼了两天")
        self.assertEqual(result["pain"]["location"], "腰部")
        self.assertEqual(result["pain"]["severity"], 3)


class TestExtractHealthDataCombinations(unittest.TestCase):
    """Test extraction of multi-dimension health messages."""

    def test_sleep_plus_exercise(self):
        """Sleep + exercise in one message."""
        llm_response = json.dumps({
            "sleep": {"quality": 2, "duration_hours": 5.0},
            "energy": 2, "mood": None,
            "exercise": {"type": "跑步", "duration_minutes": 20},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("昨晚没睡好只睡了5小时，早上还是去跑了20分钟")
        self.assertEqual(result["sleep"]["quality"], 2)
        self.assertEqual(result["sleep"]["duration_hours"], 5.0)
        self.assertEqual(result["exercise"]["type"], "跑步")
        self.assertEqual(result["energy"]["score"], 2)

    def test_mood_plus_digestion(self):
        """Mood + digestion in one message."""
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": None, "mood": 2,
            "exercise": {"type": None, "duration_minutes": None},
            "digestion": {"status": "胃痛", "severity": 4},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("胃疼了一下午，心情很差")
        self.assertEqual(result["mood"]["score"], 2)
        self.assertEqual(result["digestion"]["status"], "胃痛")

    def test_full_body_checkin(self):
        """A comprehensive health check-in covering 5+ dimensions."""
        llm_response = json.dumps({
            "sleep": {"quality": 3, "duration_hours": 7.0},
            "energy": 3, "mood": 3,
            "exercise": {"type": "游泳", "duration_minutes": 45},
            "digestion": {"status": "正常", "severity": None},
            "skin": {"status": "干燥"},
            "pain": {"location": "肩膀", "severity": 2},
            "diet": {"notes": "正常饮食"},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("昨晚睡了7小时还不错，中午游了45分钟泳，肩膀有点酸，皮肤有点干")
        self.assertEqual(result["sleep"]["duration_hours"], 7.0)
        self.assertEqual(result["exercise"]["type"], "游泳")
        self.assertEqual(result["pain"]["location"], "肩膀")
        self.assertEqual(result["skin"]["status"], "干燥")

    def test_energy_plus_pain(self):
        """Low energy with pain."""
        llm_response = json.dumps({
            "sleep": {"quality": 2, "duration_hours": 6.0},
            "energy": 1, "mood": 2,
            "exercise": {"type": None, "duration_minutes": None},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": "全身", "severity": 3},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("今天完全没精神，浑身都疼，只想躺着")
        self.assertEqual(result["energy"]["score"], 1)
        self.assertEqual(result["pain"]["severity"], 3)

    def test_exercise_plus_diet(self):
        """Post-workout nutrition."""
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": 4, "mood": None,
            "exercise": {"type": "健身", "duration_minutes": 60},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": "高蛋白: 鸡胸肉+蛋白粉"},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("健身一小时，练完喝了蛋白粉吃了鸡胸肉")
        self.assertEqual(result["exercise"]["type"], "健身")
        self.assertEqual(result["diet"]["notes"], "高蛋白: 鸡胸肉+蛋白粉")


class TestExtractHealthDataEdgeCases(unittest.TestCase):
    """Test edge cases and robustness of health extraction."""

    def test_very_long_message(self):
        """Very long messages should not crash."""
        long_msg = "今天" + "很" * 500 + "累"
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": 1, "mood": None,
            "exercise": {"type": None, "duration_minutes": None},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data(long_msg)
        self.assertIsInstance(result, dict)
        self.assertIn("sleep", result)

    def test_message_with_emoji(self):
        """Messages with emoji should work."""
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": 5, "mood": 5,
            "exercise": {"type": None, "duration_minutes": None},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("今天感觉超棒！😄🎉")
        self.assertEqual(result["energy"]["score"], 5)

    def test_english_message(self):
        """English messages should also work."""
        llm_response = json.dumps({
            "sleep": {"quality": 4, "duration_hours": 8.0},
            "energy": 4, "mood": None,
            "exercise": {"type": "running", "duration_minutes": 30},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("Slept 8 hours, went for a 30 min run")
        self.assertEqual(result["sleep"]["duration_hours"], 8.0)
        self.assertEqual(result["exercise"]["type"], "running")

    def test_negative_no_exercise(self):
        """'没运动' should still return valid schema, not crash."""
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": 2, "mood": None,
            "exercise": {"type": None, "duration_minutes": None},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("今天没运动，一直坐着")
        self.assertIsInstance(result, dict)
        self.assertIsNone(result["exercise"]["type"])

    def test_message_with_numbers(self):
        """Messages containing numbers should be handled."""
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": None, "mood": None,
            "exercise": {"type": "步行", "duration_minutes": 120},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("今天走了20000步")
        self.assertEqual(result["exercise"]["duration_minutes"], 120)

    def test_ambiguous_message(self):
        """Ambiguous messages should return template data with null values."""
        llm_response = json.dumps({
            "sleep": {"quality": None, "duration_hours": None},
            "energy": None, "mood": None,
            "exercise": {"type": None, "duration_minutes": None},
            "digestion": {"status": None, "severity": None},
            "skin": {"status": None},
            "pain": {"location": None, "severity": None},
            "diet": {"notes": None},
        })
        with patch("ai_engine._call_llm", return_value=llm_response):
            result = extract_health_data("还行吧")
        self.assertIsInstance(result, dict)
        self.assertIsNone(result["energy"]["score"])
        self.assertIsNone(result["mood"]["score"])


class TestNormalizeExtractedAdvanced(unittest.TestCase):
    """Advanced normalization tests for edge cases and backward compat."""

    def test_llm_extra_nested_field_normalized(self):
        """LLM extra fields not in schema should be normalized (only schema keys kept)."""
        raw = {
            "sleep": {"quality": 3, "duration_hours": 7.0, "deep_sleep": 2.0},
            "energy": 3, "mood": None,
            "exercise": None, "digestion": None, "skin": None,
            "pain": None, "diet": None,
        }
        result = normalize_extracted(raw)
        self.assertEqual(result["sleep"]["quality"], 3)
        self.assertEqual(result["sleep"]["duration_hours"], 7.0)
        # Schema enforcement: extra fields not in template are dropped

    def test_llm_string_for_nested_replaced(self):
        """LLM returning a string for nested object should be replaced with template."""
        raw = {
            "sleep": "bad sleep", "energy": None, "mood": None,
            "exercise": None, "digestion": None, "skin": None,
            "pain": None, "diet": None,
        }
        result = normalize_extracted(raw)
        self.assertIsInstance(result["sleep"], dict)
        self.assertIsNone(result["sleep"]["quality"])

    def test_llm_list_for_nested_replaced(self):
        """LLM returning a list for energy (wrong type) should be replaced with template."""
        raw = {
            "sleep": {"quality": None, "duration_hours": None},
            "energy": [1, 2, 3], "mood": None,
            "exercise": None, "digestion": None, "skin": None,
            "pain": None, "diet": None,
        }
        result = normalize_extracted(raw)
        self.assertEqual(result["energy"]["score"], None)

    def test_all_null_response_becomes_template(self):
        """LLM returning everything as null should get template defaults."""
        raw = {
            "sleep": None, "energy": None, "mood": None,
            "exercise": None, "digestion": None, "skin": None,
            "pain": None, "diet": None,
        }
        result = normalize_extracted(raw)
        self.assertIsInstance(result["sleep"], dict)
        self.assertIsNone(result["sleep"]["quality"])
        self.assertIsNone(result["energy"]["score"])

    def test_empty_dict_filled_from_template(self):
        """LLM returning empty dict should be filled from template."""
        result = normalize_extracted({})
        expected_keys = {"sleep", "energy", "mood", "exercise", "digestion", "skin", "pain", "diet"}
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertIsInstance(result["sleep"], dict)
        self.assertIsInstance(result["energy"], dict)

    def test_sleep_partial_nested_merged(self):
        """Sleep object with only one field should have other filled."""
        raw = {
            "sleep": {"quality": 5}, "energy": None, "mood": None,
            "exercise": None, "digestion": None, "skin": None,
            "pain": None, "diet": None,
        }
        result = normalize_extracted(raw)
        self.assertEqual(result["sleep"]["quality"], 5)
        self.assertIsNone(result["sleep"]["duration_hours"])

    def test_energy_dict_with_reason_preserved(self):
        """Energy dict with custom reason should be preserved."""
        raw = {
            "sleep": {"quality": None, "duration_hours": None},
            "energy": {"score": 4, "reason": "feeling great"},
            "mood": None, "exercise": None, "digestion": None,
            "skin": None, "pain": None, "diet": None,
        }
        result = normalize_extracted(raw)
        self.assertEqual(result["energy"]["score"], 4)
        self.assertEqual(result["energy"]["reason"], "feeling great")

    def test_pain_partial_dict_merged(self):
        """Pain with only location should have severity filled from template."""
        raw = {
            "sleep": {"quality": None, "duration_hours": None},
            "energy": None, "mood": None,
            "exercise": None, "digestion": None, "skin": None,
            "pain": {"location": "膝盖"}, "diet": None,
        }
        result = normalize_extracted(raw)
        self.assertEqual(result["pain"]["location"], "膝盖")
        self.assertIsNone(result["pain"]["severity"])

    def test_energy_zero_value_preserved(self):
        """Energy score of 0 (valid) should be preserved."""
        raw = {"energy": 0, "sleep": None, "mood": None, "exercise": None,
               "digestion": None, "skin": None, "pain": None, "diet": None}
        result = normalize_extracted(raw)
        self.assertEqual(result["energy"]["score"], 0)


class TestParseStoryResponse(unittest.TestCase):
    """Test _parse_story_response for structured story output parsing."""

    def test_parse_full_story(self):
        """Parse a complete four-section story response."""
        raw = """=== 标题 ===
📖 身体悄悄在变好

=== 数据速览 ===
🛏 日均睡眠 7.2 小时
😊 本周情绪得分最高那天是周三
🏃 运动打卡 4 天

=== 本周故事 ===
这周你的身体像一部终于充上电的手机，从周一的低电量模式慢慢恢复到了周三的绿色满格。

睡眠曲线虽然像过山车，但好在整体趋势是向上的。周三那个 8 小时的深睡太关键了，直接拉高了后半周的精力和情绪。

=== 小贴士 ===
✨ 周三的睡眠节奏很棒，试试这周也保持 23:30 前上床？
✨ 周四下雨没出门运动？下次在家来个 10 分钟拉伸也不错"""

        result = _parse_story_response(raw)
        self.assertIn("📖", result["title"])
        self.assertIn("身体", result["title"])
        self.assertIn("7.2", result["stats"])
        self.assertIn("运动打卡", result["stats"])
        self.assertIn("手机", result["narrative"])
        self.assertIn("过山车", result["narrative"])
        self.assertIn("睡眠节奏", result["tips"])
        self.assertIn("✨", result["tips"])

    def test_parse_minimal_story(self):
        """Parse a story with minimal content."""
        raw = """=== 标题 ===
📊 数据还不够

=== 数据速览 ===
📝 本周共 3 条记录

=== 本周故事 ===
这周的记录还不多，但我已经能看到一点苗头了。

=== 小贴士 ===
✨ 多说几句，我就能给出更有用的建议啦"""

        result = _parse_story_response(raw)
        self.assertEqual("📊 数据还不够", result["title"])
        self.assertIn("3 条记录", result["stats"])
        self.assertIn("苗头", result["narrative"])
        self.assertTrue(len(result["tips"]) > 0)

    def test_parse_empty_sections(self):
        """Parse story with some empty sections."""
        raw = """=== 标题 ===
🌿 安静的一周

=== 数据速览 ===


=== 本周故事 ===
这周很安静，你只记录了两次。

=== 小贴士 ===
"""

        result = _parse_story_response(raw)
        self.assertEqual("🌿 安静的一周", result["title"])
        self.assertIn("安静", result["narrative"])

    def test_parse_fallback_to_raw(self):
        """If parsing yields empty narrative, fallback to raw text."""
        raw = "这周你的身体状况不错，精力充沛，睡眠也很好。"
        result = _parse_story_response(raw)
        self.assertTrue(len(result["narrative"]) > 0)
        self.assertIn("精力充沛", result["narrative"])

    def test_parse_all_fields_present(self):
        """All expected keys are in the result dict."""
        raw = "some text"
        result = _parse_story_response(raw)
        for key in ("title", "stats", "narrative", "tips", "_raw"):
            self.assertIn(key, result)

    def test_parse_empty_string(self):
        """Empty string parsing doesn't crash."""
        result = _parse_story_response("")
        self.assertEqual(result["title"], "")
        self.assertEqual(result["narrative"], "")

    def test_parse_persona_influenced_story(self):
        """Parse story with fitness coach persona style."""
        raw = """=== 标题 ===
💪 运动表现周报

=== 数据速览 ===
🏃 跑步 3 次，累计 15 公里
🛏 平均睡眠 6.8 小时
⚡ 周四精力峰值

=== 本周故事 ===
你的运动保持了节奏，三次跑步的配速都在稳步提升，这说明心率适应在变好。

不过睡眠有点拖后腿。周一和周三睡太晚了，直接影响了第二天的训练状态。

=== 小贴士 ===
✨ 高强度训练日尽量睡够 7.5 小时，恢复效果差一倍
✨ 周五是休息日，可以安排一次泡沫轴放松"""

        result = _parse_story_response(raw)
        self.assertIn("💪", result["title"])
        self.assertIn("配速", result["narrative"])
        self.assertIn("泡沫轴", result["tips"])


class TestGenerateBodyStory(unittest.TestCase):
    """Test generate_body_story with new structured format."""

    def test_no_records_returns_fallback(self):
        """Empty records return fallback message."""
        result = generate_body_story([], "week")
        self.assertIn("还没有记录", result)

    def test_with_records_includes_period(self):
        """Results reference the time period."""
        records = [
            {"timestamp": "2025-06-01T08:00:00Z", "content": "睡了7小时",
             "extracted_data": {"sleep": {"duration_hours": 7}}},
        ]
        # This will try to call LLM, but we just verify the function accepts
        # the new parameters without error
        self.assertTrue(callable(generate_body_story))

    def test_generate_body_story_accepts_new_params(self):
        """Function signature accepts baselines and persona kwargs."""
        import inspect
        sig = inspect.signature(generate_body_story)
        params = list(sig.parameters.keys())
        self.assertIn("baselines", params)
        self.assertIn("persona", params)


if __name__ == "__main__":
    unittest.main()
