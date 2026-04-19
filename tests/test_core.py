import os
import shutil
import tempfile
import time
import unittest
import urllib.error
from unittest.mock import patch

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.autofix import AutoFixAgent, parse_autofix_plan
from agent.review import ReviewAgent
from agent.code_graph import CodeGraphIndex
from agent.context_manager import ContextManager, estimate_tokens, cosine_similarity, extract_python_symbols
from agent.core import Agent
from agent.patch_engine import PatchEngine, parse_patch_plan
from agent.session_memory import SessionMemory
from agent.output_schema import parse_answer_schema
from llm.client import LLMClient
from mcp_server.retrieval_index import RetrievalIndex
from mcp_server.tools import list_files, read_file
from utils.safety import validate_command_safety
from utils.logger import TraceLogger
from config import MAX_FILE_BYTES


class StubEmbeddingLLM:
    def __init__(self):
        self.embedding_calls = 0

    def get_embedding(self, _text):
        self.embedding_calls += 1
        return [1.0, 0.0, 0.0]


class TestCoreUtilities(unittest.TestCase):
    def test_estimate_tokens(self):
        self.assertEqual(estimate_tokens("word"), 1)
        sample = "hello, world! this is a test."
        self.assertEqual(estimate_tokens(sample), len(sample.split()) + (3 // 2))

    def test_cosine_similarity(self):
        self.assertEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine_similarity([1, 0], []), 0.0)
        self.assertEqual(cosine_similarity([0, 0], [1, 0]), 0.0)

    def test_symbol_extraction(self):
        content = "class Alpha:\n    pass\n\ndef beta():\n    return 1\n"
        symbols = extract_python_symbols(content)
        self.assertIn("alpha", symbols)
        self.assertIn("beta", symbols)

    def test_output_schema(self):
        parsed = parse_answer_schema("Answer:\nYes\n\nJustification:\n- A\n- B")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.answer, "Yes")
        self.assertEqual(len(parsed.justification), 2)


class TestMCPTools(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_list_files_pruning(self):
        ok_path = os.path.join(self.test_dir, "test.py")
        with open(ok_path, "w", encoding="utf-8") as f:
            f.write("print('ok')")

        ignore_dir = os.path.join(self.test_dir, "node_modules")
        os.makedirs(ignore_dir)
        with open(os.path.join(ignore_dir, "test.js"), "w", encoding="utf-8") as f:
            f.write("ignore me")

        bad_ext_path = os.path.join(self.test_dir, "test.exe")
        with open(bad_ext_path, "wb") as f:
            f.write(b"0" * 10)

        big_path = os.path.join(self.test_dir, "big.py")
        with open(big_path, "wb") as f:
            f.write(b"0" * (MAX_FILE_BYTES + 10))

        files, stats = list_files(self.test_dir)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["path"], ok_path)
        self.assertEqual(stats["skipped_ext"], 1)
        self.assertEqual(stats["skipped_size"], 1)

    def test_read_file_caching_and_error(self):
        test_file = os.path.join(self.test_dir, "cache_test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("V1")
        self.assertEqual(read_file(test_file), "V1")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("V2")
        os.utime(test_file, (time.time() + 1, time.time() + 1))
        self.assertEqual(read_file(test_file), "V2")
        self.assertIsNone(read_file(os.path.join(self.test_dir, "missing.txt")))


class TestRetrievalAndGraph(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.lexical_db = os.path.join(self.test_dir, "lex.sqlite")
        self.graph_db = os.path.join(self.test_dir, "graph.sqlite")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_retrieval_index_upsert_and_query(self):
        idx = RetrievalIndex(self.lexical_db)
        idx.upsert_files(
            [
                {"path": "/tmp/a.py", "mtime": 1.0, "text_content": "auth login token refresh"},
                {"path": "/tmp/b.py", "mtime": 1.0, "text_content": "render component layout"},
            ]
        )
        out = idx.query("auth OR token", limit=5)
        self.assertTrue(any(path == "/tmp/a.py" for path, _ in out))

    def test_code_graph_boosts(self):
        graph = CodeGraphIndex(self.graph_db)
        graph.upsert_python_files(
            [
                {
                    "path": "/tmp/a.py",
                    "mtime": 1.0,
                    "content": "import json\n\ndef auth_token():\n    return make_token()\n\ndef make_token():\n    return 'x'\n",
                }
            ]
        )
        boosts = graph.query_file_boosts(["auth_token", "json"])
        self.assertIn("/tmp/a.py", boosts)
        self.assertGreater(boosts["/tmp/a.py"], 0.0)


class TestLLMClient(unittest.TestCase):
    def test_select_best_model(self):
        llm = LLMClient()
        llm._groq_models = ["llama-3.3-70b-versatile", "gemma-7b"]
        llm._openrouter_models = ["deepseek/deepseek-chat", "gpt-4"]
        self.assertEqual(llm.select_best_model("fast"), ("groq", "llama-3.3-70b-versatile"))
        self.assertEqual(llm.select_best_model("strong"), ("openrouter", "deepseek/deepseek-chat"))

    def test_query_model_fallback_sets_error_metadata(self):
        llm = LLMClient()
        with patch.object(
            llm,
            "_execute_query",
            side_effect=[Exception("provider down 1"), Exception("provider down 2"), Exception("provider down 3"), "local answer"],
        ):
            result = llm.query_model("openrouter", "demo-model", "hello")
        self.assertEqual(result, "local answer")
        err = llm.consume_last_error()
        self.assertIsNotNone(err)
        self.assertEqual(err["type"], "fallback_used")

    def test_rate_limit_retry(self):
        llm = LLMClient()
        rate_err = urllib.error.HTTPError(url="", code=429, msg="rate", hdrs=None, fp=None)
        with patch.object(llm, "_execute_query", side_effect=[rate_err, "ok"]):
            result = llm.query_model("openrouter", "demo-model", "hello")
        self.assertEqual(result, "ok")
        usage = llm.consume_last_usage()
        self.assertIsNotNone(usage)
        self.assertGreater(usage["total_tokens"], 0)


class TestContextManagerModes(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "alpha.py")
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("class Handler:\n    pass\n\ndef auth_token():\n    return 'ok'\n")
        self.file_infos = [{"path": self.test_file, "mtime": os.path.getmtime(self.test_file)}]

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_embeddings_disabled_skips_embedding_calls(self):
        llm = StubEmbeddingLLM()
        with patch("agent.context_manager.ENABLE_EMBEDDINGS", False):
            cm = ContextManager(llm)
            chunks, trace = cm.rank_and_extract(self.file_infos, "auth token")
        self.assertEqual(llm.embedding_calls, 0)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("selected_files", trace)

    def test_embeddings_enabled_calls_embedding(self):
        llm = StubEmbeddingLLM()
        with patch("agent.context_manager.ENABLE_EMBEDDINGS", True):
            cm = ContextManager(llm)
            chunks, _trace = cm.rank_and_extract(self.file_infos, "auth token")
        self.assertGreaterEqual(llm.embedding_calls, 1)
        self.assertGreaterEqual(len(chunks), 1)


class TestPatchEngineAndSafety(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.file_path = os.path.join(self.test_dir, "sample.py")
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write("def add(a, b):\n    return a+b\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_patch_plan_parse_and_apply_snippet(self):
        raw = (
            "{"
            '"summary":"fix spacing",'
            '"done":false,'
            '"risk":"low",'
            '"commit_message":"athena: format add",'
            '"operations":[{"type":"replace_snippet","file":"sample.py","find":"return a+b","replace":"return a + b","reason":"pep8"}],'
            '"validation_commands":["python3 -m unittest discover -s tests -p test*.py -v"]'
            "}"
        )
        plan = parse_patch_plan(raw)
        self.assertIsNotNone(plan)
        engine = PatchEngine(self.test_dir)
        dry = engine.apply_plan(plan, dry_run=True)
        self.assertTrue(dry["ok"])
        real = engine.apply_plan(plan, dry_run=False)
        self.assertTrue(real["ok"])
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("return a + b", content)

    def test_patch_engine_replace_symbol(self):
        raw = (
            "{"
            '"summary":"rewrite symbol",'
            '"done":false,'
            '"risk":"low",'
            '"commit_message":"athena: rewrite function",'
            '"operations":[{"type":"replace_symbol","file":"sample.py","symbol":"add","symbol_kind":"function","new_code":"def add(a, b):\\n    return a + b\\n","reason":"normalize"}],'
            '"validation_commands":[]'
            "}"
        )
        plan = parse_patch_plan(raw)
        self.assertIsNotNone(plan)
        engine = PatchEngine(self.test_dir)
        result = engine.apply_plan(plan, dry_run=False)
        self.assertTrue(result["ok"])
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("return a + b", content)

    def test_safety_blocks_dangerous_command(self):
        ok, _ = validate_command_safety("python3 -m unittest discover -s tests -p test*.py -v")
        self.assertTrue(ok)
        blocked, reason = validate_command_safety("rm -rf /")
        self.assertFalse(blocked)
        self.assertIn("blocked term", reason)

    def test_checkpoint_create_restore(self):
        engine = PatchEngine(self.test_dir)
        checkpoint = engine.create_checkpoint(["sample.py"], reason="pre-change")
        self.assertTrue(checkpoint["ok"])
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write("def add(a, b):\n    return a - b\n")
        restore = engine.restore_checkpoint(checkpoint["checkpoint_id"])
        self.assertTrue(restore["ok"])
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("return a+b", content)

    def test_patch_engine_create_and_delete_file(self):
        engine = PatchEngine(self.test_dir)
        raw = (
            "{"
            '"summary":"create file",'
            '"done":false,'
            '"risk":"low",'
            '"commit_message":"athena: create tmp",'
            '"operations":[{"type":"create_file","file":"new_file.py","new_code":"def x():\\n    return 1\\n","reason":"add module"}],'
            '"validation_commands":[]'
            "}"
        )
        plan = parse_patch_plan(raw)
        self.assertIsNotNone(plan)
        result = engine.apply_plan(plan, dry_run=False)
        self.assertTrue(result["ok"])
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "new_file.py")))

        raw_delete = (
            "{"
            '"summary":"delete file",'
            '"done":false,'
            '"risk":"low",'
            '"commit_message":"athena: delete tmp",'
            '"operations":[{"type":"delete_file","file":"new_file.py","reason":"cleanup"}],'
            '"validation_commands":[]'
            "}"
        )
        del_plan = parse_patch_plan(raw_delete)
        self.assertIsNotNone(del_plan)
        del_result = engine.apply_plan(del_plan, dry_run=False)
        self.assertTrue(del_result["ok"])
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "new_file.py")))

    def test_patch_engine_append_snippet(self):
        engine = PatchEngine(self.test_dir)
        raw = (
            "{"
            '"summary":"append snippet",'
            '"done":false,'
            '"risk":"low",'
            '"commit_message":"athena: append",'
            '"operations":[{"type":"append_snippet","file":"sample.py","find":"return a+b","new_code":"# appended note","reason":"annotate"}],'
            '"validation_commands":[]'
            "}"
        )
        plan = parse_patch_plan(raw)
        self.assertIsNotNone(plan)
        result = engine.apply_plan(plan, dry_run=False)
        self.assertTrue(result["ok"])
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("# appended note", content)


class TestSessionMemory(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_session_memory_roundtrip(self):
        mem = SessionMemory(self.test_dir, memory_file="mem.json")
        mem.add_entry("q1", "summary1", "completed", ["a.py"], "high")
        mem2 = SessionMemory(self.test_dir, memory_file="mem.json")
        recent = mem2.recent_entries(1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["query"], "q1")


class TestLoggerExport(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_export_jsonl(self):
        logger = TraceLogger()
        logger.log("event_a", {"k": 1})
        target = os.path.join(self.test_dir, "trace.jsonl")
        out = logger.export_jsonl(target)
        self.assertEqual(out, os.path.abspath(target))
        with open(out, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 1)


class TestAgentIntegration(unittest.TestCase):
    def test_no_context_path_sets_low_confidence(self):
        agent = Agent()
        agent.llm.route_model = lambda _task, quality="balanced": ("local", "llama3")

        async def fake_query_async(_provider, _model, prompt, _system=""):
            if "Evaluate for hallucination" in prompt:
                return "APPROVE: expected due to no context"
            return "Answer:\ninsufficient context\n\nJustification:\n- insufficient context\n- no files"

        agent.llm.query_model_async = fake_query_async
        agent.llm.consume_last_error = lambda: None
        agent.llm.consume_last_usage = lambda: {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22, "total_cost": 0.0001}

        with patch("agent.core.list_files", return_value=([], {"total_scanned": 0})):
            with patch.object(
                agent.context_manager,
                "rank_and_extract",
                return_value=([], {"query_intent": "search", "estimated_tokens_used": 0, "execution_times": {}, "selected_files": []}),
            ):
                result = agent.run("debug", "where is auth?", self._temp_workspace())

        self.assertIn("insufficient context", result.lower())
        eval_event = next(x for x in agent.logger.trace if x["event"] == "evaluation_results")
        self.assertEqual(eval_event["details"]["final_confidence"], "Low")
        self.assertTrue(eval_event["details"]["schema_valid"])

    def _temp_workspace(self) -> str:
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(temp_dir))
        return temp_dir


class TestAutoFix(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "sample.py")
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("def add(a, b):\n    return a+b\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_parse_autofix_plan_alias(self):
        raw = (
            "{"
            '"summary":"fix spacing",'
            '"done":false,'
            '"risk":"low",'
            '"commit_message":"athena: style",'
            '"operations":[{"type":"replace_snippet","file":"sample.py","find":"return a+b","replace":"return a + b","reason":"style"}],'
            '"validation_commands":[]'
            "}"
        )
        plan = parse_autofix_plan(raw)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.summary, "fix spacing")
        self.assertEqual(len(plan.operations), 1)

    def test_autofix_high_risk_requires_approval(self):
        agent = Agent()
        fixer = AutoFixAgent(agent)

        async def fake_create_plan(_query, _context, _feedback, _memory):
            return parse_patch_plan(
                "{"
                '"summary":"dangerous edit",'
                '"done":false,'
                '"risk":"high",'
                '"commit_message":"athena: risky",'
                '"operations":[{"type":"replace_snippet","file":"sample.py","find":"return a+b","replace":"return a + b","reason":"style"}],'
                '"validation_commands":[]'
                "}"
            )

        fixer._create_plan = fake_create_plan
        with patch.object(fixer, "_build_context", return_value=("ctx", {})):
            report = fixer.run(
                query="fix style",
                path=self.test_dir,
                apply=True,
                approve_high_risk=False,
                max_iterations=1,
            )
        self.assertEqual(report["status"], "blocked")

    def test_autofix_detects_validation_commands(self):
        agent = Agent()
        fixer = AutoFixAgent(agent)
        with open(os.path.join(self.test_dir, "package.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        with open(os.path.join(self.test_dir, "go.mod"), "w", encoding="utf-8") as f:
            f.write("module x")
        commands = fixer._detect_validation_commands(self.test_dir)
        self.assertIn("npm test", commands)
        self.assertIn("go test ./...", commands)


class TestReviewAgent(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_review_no_repo(self):
        agent = Agent()
        reviewer = ReviewAgent(agent)
        report = reviewer.run(path=self.test_dir)
        self.assertEqual(report["status"], "no_diff")


if __name__ == "__main__":
    unittest.main()
