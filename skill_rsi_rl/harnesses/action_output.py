"""Decode explicit compatible-provider tool markup at the public action boundary."""
import json
import re


def explicit_past_action(text):
    """Preserve the tool names and JSON arguments of an explicit DSML call."""
    if '<｜｜DSML｜｜' not in text:
        return None
    calls = []
    for name, body in re.findall(r'<｜｜DSML｜｜invoke name="([^"]+)">(.*?)</｜｜DSML｜｜invoke>', text, re.S):
        # Some providers put whitespace after the DSML delimiter.
        arguments = re.findall(r'<｜｜DSML｜｜parameter name="arguments" string="(?:false|true)">(.*?)</｜｜DSML｜｜parameter>', body, re.S)
        if len(arguments) != 1:
            raise ValueError('explicit DSML tool call needs one JSON arguments object')
        value = json.loads(arguments[0])
        if not isinstance(value, dict):
            raise ValueError('explicit tool arguments must be an object')
        calls.append(dict(name=name, arguments=value))
    if not calls:
        # Accept the spelling used by the public API compatibility transport.
        normalized = re.sub(r'(<\/?｜｜DSML｜｜)\s+', r'\1', text)
        if normalized != text:
            return explicit_past_action(normalized)
        raise ValueError('DSML contains no supported explicit tool call')
    return dict(tool_calls=calls, final_output=None)
