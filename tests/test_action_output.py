import unittest

from skill_rsi_rl.harnesses.action_output import explicit_past_action


class ActionOutputTests(unittest.TestCase):
    def test_actual_provider_call_preserves_name_and_arguments(self):
        text = ('<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="notes_list">\n'
                '<｜｜DSML｜｜ parameter name="arguments" string="true">{"tags": []}'
                '</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>')
        expected = dict(tool_calls=[dict(name='notes_list', arguments={'tags':[]})], final_output=None)
        self.assertEqual(explicit_past_action(text), expected)
        self.assertEqual(explicit_past_action(text.replace('string="true"','string="false"')), expected)

    def test_plain_json_uses_native_decoder(self):
        self.assertIsNone(explicit_past_action('{"tool_calls":[],"final_output":"ok"}'))

    def test_malformed_explicit_call_is_rejected(self):
        for text in ('<｜｜DSML｜｜ calls></｜｜DSML｜｜ calls>',
                     '<｜｜DSML｜｜ invoke name="notes_list"></｜｜DSML｜｜ invoke>'):
            with self.assertRaises(ValueError):
                explicit_past_action(text)
