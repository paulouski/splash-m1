import json
import tracemalloc
import unittest

from jsonschema import Draft202012Validator

from dev.tests.test_server import (
    FakeRuntime,
    FakeTokenizer,
    Harness,
    Plan,
    _byte_backend,
)
from server import api_shapes
from server import output as model_output
from server import server as api
from server.tool_schema import ToolPolicy

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
TOOL = {"type": "function", "function": {"name": "weather", "parameters": SCHEMA}}


def events(payload):
    return [
        json.loads(line[6:])
        for line in payload.decode().splitlines()
        if line.startswith("data: {")
    ]


class PartialToolOutputTests(unittest.TestCase):
    def test_prefix_scanner_handles_nested_containers_and_scalar_boundaries(self):
        values = [
            [],
            {},
            [[], {}, [True, None]],
            {"a": [{"b": [1, 2]}]},
            -1.25e-19,
            'a\\"\n北京😀',
            True,
            None,
        ]
        for value in values:
            text = json.dumps(value)
            for end in range(len(text) + 1):
                prefix = text[:end]
                try:
                    json.loads(prefix)
                    incomplete = False
                except ValueError:
                    incomplete = True
                with self.subTest(prefix=prefix):
                    self.assertEqual(api_shapes._unfinished_json(prefix), incomplete)
        for invalid in (
            "[1,]",
            '{"a":1,}',
            "{1:",
            "[1 2",
            "1.2.",
            "1e2e",
            "1e2.",
            "01",
            "true false",
            '"\\u0z',
            '"\x01',
            '{"a" 1',
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(api_shapes._unfinished_json(invalid))

    def test_flat_partial_arguments_do_not_allocate_per_element(self):
        # Input construction is outside the measurement. The prefix scanner
        # should retain nesting state, not an array of parsed values/matches.
        arguments = '{"values":[' + ",".join("1" for _ in range(20000)) + ","
        tracemalloc.start()
        try:
            self.assertTrue(api_shapes._unfinished_json(arguments))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 64 * 1024)

    def harness(self, *plans):
        value = Harness(FakeRuntime(*plans))
        self.addCleanup(value.close)
        return value

    def test_tool_only_whitespace_is_absent_in_every_protocol(self):
        call = (
            "<tool_call>\n<function=weather>\n<parameter=city>\n"
            "Paris\n</parameter>\n</function>\n</tool_call>"
        )
        for path in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
            for stream in (False, True):
                for thinking in (False, True):
                    with self.subTest(path=path, stream=stream, thinking=thinking):
                        text = (
                            ("reason</think>" if thinking else "")
                            + " \n"
                            + call
                            + "\n\t"
                            + call
                            + "\n"
                        )
                        tokenizer = FakeTokenizer()
                        tokenizer.fragments = dict(enumerate(dict.fromkeys(text), 1))
                        token_ids = {
                            value: key for key, value in tokenizer.fragments.items()
                        }
                        tokenizer.backend_tokenizer = _byte_backend(tokenizer.fragments)
                        harness = Harness(
                            FakeRuntime(Plan([[token_ids[char]] for char in text])),
                            tokenizer=tokenizer,
                            max_context=8192,
                        )
                        try:
                            body = {"model": "test-model", "stream": stream}
                            if path == "/v1/responses":
                                body.update(
                                    input="Paris",
                                    reasoning={
                                        "effort": "high" if thinking else "none"
                                    },
                                    tools=[{"type": "function", **TOOL["function"]}],
                                )
                            else:
                                body.update(
                                    messages=[{"role": "user", "content": "Paris"}],
                                    max_tokens=4096,
                                )
                                if path == "/v1/messages":
                                    body.update(
                                        thinking={
                                            "type": "enabled",
                                            "budget_tokens": 2048,
                                        }
                                        if thinking
                                        else {"type": "disabled"},
                                        tools=[
                                            {"name": "weather", "input_schema": SCHEMA}
                                        ],
                                    )
                                else:
                                    body.update(
                                        reasoning_effort="high" if thinking else "none",
                                        tools=[TOOL],
                                    )
                            status, _, payload = harness.request("POST", path, body)
                            self.assertEqual(status, 200, payload)
                            if stream:
                                rows = events(payload)
                                self.assertFalse(
                                    any("error" in row for row in rows), rows
                                )
                                if path == "/v1/chat/completions":
                                    deltas = [
                                        row["choices"][0]["delta"]
                                        for row in rows
                                        if row.get("choices")
                                    ]
                                    self.assertFalse(
                                        any(delta.get("content") for delta in deltas)
                                    )
                                    self.assertEqual(
                                        sum(
                                            "name" in call["function"]
                                            for delta in deltas
                                            for call in delta.get("tool_calls", [])
                                        ),
                                        2,
                                    )
                                elif path == "/v1/responses":
                                    self.assertFalse(
                                        any(
                                            row["type"] == "response.output_text.delta"
                                            for row in rows
                                        )
                                    )
                                    self.assertEqual(
                                        [
                                            item["type"]
                                            for item in rows[-1]["response"]["output"]
                                            if item["type"] != "reasoning"
                                        ],
                                        ["function_call", "function_call"],
                                    )
                                else:
                                    blocks = [
                                        row["content_block"]
                                        for row in rows
                                        if row["type"] == "content_block_start"
                                    ]
                                    self.assertEqual(
                                        [
                                            block["type"]
                                            for block in blocks
                                            if block["type"] != "thinking"
                                        ],
                                        ["tool_use", "tool_use"],
                                    )
                            else:
                                result = json.loads(payload)
                                if path == "/v1/chat/completions":
                                    message = result["choices"][0]["message"]
                                    self.assertIsNone(message["content"])
                                    self.assertEqual(len(message["tool_calls"]), 2)
                                elif path == "/v1/responses":
                                    self.assertEqual(
                                        [
                                            item["type"]
                                            for item in result["output"]
                                            if item["type"] != "reasoning"
                                        ],
                                        ["function_call", "function_call"],
                                    )
                                else:
                                    self.assertEqual(
                                        [
                                            item["type"]
                                            for item in result["content"]
                                            if item["type"] != "thinking"
                                        ],
                                        ["tool_use", "tool_use"],
                                    )
                        finally:
                            harness.close()

    def test_projector_preserves_whitespace_without_a_tool(self):
        for text in (" \n\t", " \nhello \t\n"):
            for incomplete in (False, True):
                projector = model_output.StreamingToolCallProjector(None, "whitespace")
                emitted = []
                for character in text:
                    emitted.extend(
                        value
                        for kind, value in projector.put(character)
                        if kind == "content"
                    )
                canonical, calls = (
                    projector.interrupted_result() if incomplete else (text, [])
                )
                emitted.extend(projector.finish(canonical, calls, incomplete))
                self.assertEqual(canonical, text)
                self.assertEqual("".join(emitted), text)

    def test_text_after_a_call_streams_and_survives_a_cut(self):
        policy = ToolPolicy(
            {"weather": Draft202012Validator(SCHEMA)}, {"weather": SCHEMA}, False, True
        )
        call = (
            "<tool_call>\n<function=weather>\n<parameter=city>\n"
            "Paris\n</parameter>\n</function>\n</tool_call>"
        )
        projector = model_output.StreamingToolCallProjector(policy, "after-call")

        before = projector.put("Checking.\n" + call)
        self.assertEqual(
            [value for kind, value in before if kind == "content"], ["Checking.\n"]
        )
        after = projector.put("\nThe answer is 42.")
        self.assertEqual(
            [value for kind, value in after if kind == "content"],
            ["\nThe answer is 42."],
        )

        canonical_content, canonical_calls = model_output.parse_tool_calls(
            "Checking.\n" + call + "\nThe answer is 42.", "after-call", policy
        )
        self.assertEqual(canonical_content, "Checking.\n\nThe answer is 42.")
        content, calls = projector.interrupted_result()
        self.assertEqual((content, calls), (canonical_content, canonical_calls))
        self.assertEqual(
            "".join(value for kind, value in before + after if kind == "content"),
            canonical_content,
        )
        self.assertEqual(
            projector.finish(canonical_content, canonical_calls, incomplete=False), []
        )

    def test_call_framing_whitespace_is_trimmed_at_output_edges(self):
        policy = ToolPolicy(
            {"weather": Draft202012Validator(SCHEMA)}, {"weather": SCHEMA}, False, True
        )
        call = (
            "<tool_call>\n<function=weather>\n<parameter=city>\n"
            "Paris\n</parameter>\n</function>\n</tool_call>"
        )
        cases = (
            (" \n" + call + "\nAnswer", "\nAnswer"),
            (
                "Before\n" + call + "\nBetween\n" + call + "\nAfter",
                "Before\n\nBetween\n\nAfter",
            ),
            ("Before" + call + "\n \t", "Before"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                canonical_content, canonical_calls = model_output.parse_tool_calls(
                    text, "framing", policy
                )
                self.assertEqual(canonical_content, expected)
                for incomplete in (False, True):
                    with self.subTest(incomplete=incomplete):
                        projector = model_output.StreamingToolCallProjector(
                            policy, "framing"
                        )
                        streamed = []
                        for character in text:
                            streamed.extend(
                                value
                                for kind, value in projector.put(character)
                                if kind == "content"
                            )
                        if incomplete:
                            self.assertEqual(
                                projector.interrupted_result(),
                                (canonical_content, canonical_calls),
                            )
                        streamed.extend(
                            projector.finish(
                                canonical_content,
                                canonical_calls,
                                incomplete=incomplete,
                            )
                        )
                        self.assertEqual("".join(streamed), canonical_content)

    def test_partial_tool_call_marker_stays_held_after_visible_text(self):
        projector = model_output.StreamingToolCallProjector(None, "partial-marker")
        events = projector.put("Answer<tool_cal")
        self.assertEqual(
            [value for kind, value in events if kind == "content"], ["Answer"]
        )
        self.assertEqual(projector.interrupted_result(), ("Answer", []))

    def test_every_json_prefix_can_be_returned_in_tool_history(self):
        objects = [
            {"city": '北京😀\\"\n'},
            {"a": True, "b": False, "c": None, "d": [-1.25e-19, 2, {"e": 0}]},
        ]
        for value in objects:
            for ascii_only in (True, False):
                text = json.dumps(value, ensure_ascii=ascii_only)
                # An empty string is a complete call without arguments.
                for end in range(1, len(text)):
                    arguments = text[:end]
                    with self.subTest(arguments=arguments):
                        call = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call",
                                    "type": "function",
                                    "function": {
                                        "name": "weather",
                                        "arguments": arguments,
                                    },
                                }
                            ],
                        }
                        normalized = api_shapes.normalize_messages(
                            [
                                call,
                                {"role": "user", "content": "Continue"},
                            ],
                            vision=True,
                        )
                        actual = normalized[0]["tool_calls"][0]["function"]
                        self.assertEqual(actual["arguments"], arguments)
        for invalid in (
            '{"a":01',
            '{"a":"\\q',
            '{"a":true garbage',
            '{"a":,',
            '{"a":1,}',
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(api_shapes._unfinished_json(invalid))

    def test_chat_preserves_same_partial_arguments_in_both_modes(self):
        harness = self.harness(
            Plan([[13]], reason="length"), Plan([[13]], reason="length")
        )
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [TOOL],
            "reasoning_effort": "none",
            "max_tokens": 16,
        }
        status, _, raw = harness.request("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 200, raw)
        choice = json.loads(raw)["choices"][0]
        self.assertEqual(choice["finish_reason"], "length")
        arguments = choice["message"]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(arguments, '{"city":"Par')
        with self.assertRaises(ValueError):
            json.loads(arguments)
        status, _, raw = harness.request(
            "POST", "/v1/chat/completions", {**body, "stream": True}
        )
        self.assertEqual(status, 200, raw)
        chunks = events(raw)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        streamed = "".join(
            call.get("function", {}).get("arguments", "")
            for item in chunks
            for call in item["choices"][0]["delta"].get("tool_calls", [])
        )
        self.assertEqual(streamed, arguments)

    def test_responses_partial_history_is_retained_and_continues(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                harness = self.harness(Plan([[13]], reason="length"), Plan([[4]]))
                body = {
                    "model": "test-model",
                    "input": "weather",
                    "stream": stream,
                    "reasoning": {"effort": "none"},
                    "tools": [{"type": "function", **TOOL["function"]}],
                    "max_output_tokens": 16,
                }
                status, _, raw = harness.request("POST", "/v1/responses", body)
                self.assertEqual(status, 200, raw)
                response = events(raw)[-1]["response"] if stream else json.loads(raw)
                self.assertEqual(response["status"], "incomplete")
                call = response["output"][0]
                self.assertEqual(call["type"], "function_call")
                self.assertEqual(call["status"], "incomplete")
                self.assertEqual(call["arguments"], '{"city":"Par')
                status, _, raw = harness.request(
                    "POST",
                    "/v1/responses",
                    {
                        "model": "test-model",
                        "previous_response_id": response["id"],
                        "input": "Continue without that unfinished tool call.",
                        "reasoning": {"effort": "none"},
                        "max_output_tokens": 16,
                    },
                )
                self.assertEqual(status, 200, raw)
                self.assertEqual(json.loads(raw)["status"], "completed")
                status, _, stored = harness.request(
                    "GET", "/v1/responses/" + response["id"]
                )
                self.assertEqual(status, 200, stored)
                self.assertEqual(json.loads(stored)["output"], response["output"])
                history = harness.tokenizer.templates[-1][0]
                unfinished = [
                    json.loads(item["content"].split("\n", 1)[1])
                    for item in history
                    if str(item.get("content")).startswith("Incomplete tool call:\n")
                ]
                self.assertEqual(
                    unfinished, [{"name": "weather", "arguments": '{"city":"Par'}]
                )

    def test_anthropic_preserves_closed_calls_without_fabricating_partial_input(self):
        for tokens, expected in (([[13]], []), ([[5], [13]], [{"city": "Paris"}])):
            with self.subTest(tokens=tokens):
                harness = self.harness(Plan(tokens, reason="length"))
                status, _, raw = harness.request(
                    "POST",
                    "/v1/messages",
                    {
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "weather"}],
                        "thinking": {"type": "disabled"},
                        "max_tokens": 16,
                        "tools": [{"name": "weather", "input_schema": SCHEMA}],
                    },
                )
                self.assertEqual(status, 200, raw)
                response = json.loads(raw)
                self.assertEqual(response["stop_reason"], "max_tokens")
                self.assertTrue(response["content"])
                self.assertEqual(
                    [
                        item["input"]
                        for item in response["content"]
                        if item["type"] == "tool_use"
                    ],
                    expected,
                )

    def test_complete_and_partial_calls_survive_arbitrary_chunk_boundaries(self):
        tokenizer = FakeTokenizer()
        text = tokenizer.fragments[5] + tokenizer.fragments[13]
        policy = ToolPolicy(
            {"weather": Draft202012Validator(SCHEMA)}, {"weather": SCHEMA}, False, True
        )
        expected = None
        for size in (1, 3, 17, len(text)):
            projector = model_output.StreamingToolCallProjector(policy, "owned")
            for offset in range(0, len(text), size):
                projector.put(text[offset : offset + size])
            result = projector.interrupted_result()
            if expected is None:
                expected = result
            self.assertEqual(result, expected)
        self.assertEqual(
            [call["function"]["arguments"] for call in expected[1]],
            ['{"city":"Paris"}', '{"city":"Par'],
        )

    def test_closed_calls_still_require_schema_validation_at_length(self):
        schema = {
            **SCHEMA,
            "properties": {"city": {"type": "string", "enum": ["Berlin"]}},
        }
        policy = ToolPolicy(
            {"weather": Draft202012Validator(schema)}, {"weather": schema}, False, True
        )
        projector = model_output.StreamingToolCallProjector(policy, "owned")
        with self.assertRaises(api.APIError):
            projector.put(FakeTokenizer().fragments[5])
