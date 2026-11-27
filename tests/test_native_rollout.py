"""Synthetic boundary regressions; these do not claim benchmark acceptance."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

if sys.version_info < (3, 13) or importlib.util.find_spec('litellm') is None:
    raise unittest.SkipTest('native runner tests require the Python 3.13 benchmark environment')

from skill_rsi_rl.contracts import digest, propose_options
from skill_rsi_rl.harnesses import native_runner, native_spec
from skill_rsi_rl.harnesses.memory_policy import MemoryPolicy
from skill_rsi_rl.harnesses.native_history import validate_history
from skill_rsi_rl.harnesses.runner import run_rollout
from skill_rsi_rl.rollout_batch import check_group


def benchmark():
    plan = []
    for i, (mode, load, save, persist) in enumerate([
            ('continue', '', 'learned', True), ('continue', '', '', True),
            ('from_anchor', 'learned', '', True), ('fresh', '', '', False)]):
        plan.append(dict(instance_index=i, task=f'task-{i}', label=f'episode-{i}', family_id='family',
            history_mode=mode, history_load_anchor=load, history_save_anchor=save,
            requires_fresh_session=mode != 'continue', persistence_allowed=persist,
            preseed_artifacts_dir='', initial_home_fixture_dir=''))
    return dict(kind='past-bench', task='past-bench', harness='codex', split='train',
        system_params=dict(backend='skill_rsi', model='synthetic'), initial_memory={'hint': 'public context'},
        native=dict(revision='synthetic-native-revision', plan=plan))


class Executor:
    instances = []

    def __init__(self, **kwargs):
        self.restored = []
        self.session, self.previous_usage = None, None
        self.closed = False
        self.instances.append(self)

    def snapshot(self, path):
        path.mkdir()
        return digest(str(path))

    def restore(self, path, *, fresh_session):
        self.restored.append((path.name, fresh_session))
        return digest(str(path))

    def respond(self, query):
        self.session = 'synthetic-thread'
        return SimpleNamespace(action=query.response_schema(tool_calls=[], final_output='done'), metadata={})

    def close(self):
        self.closed = True

    def get_run_artifacts(self):
        return {'plugins': [], 'extensions': []}


class Session:
    reward = 0.0

    def __init__(self, config, episode, output):
        self.episode = episode

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def request(self, method, **kwargs):
        if method == 'reset':
            return dict(done=False, current=dict(instruction='Use public context',
                tool_results=[dict(content='public feedback ' + self.episode['label'])]))
        if method == 'step':
            return dict(done=True, current=None)
        return dict(reward=self.reward, success=bool(self.reward), raw_metric_name='fixture')


def policy(config, identity, seed, sample):
    selected = MemoryPolicy(dict(endpoint='http://localhost:1/sample', model='synthetic', checkpoint=identity),
                            initial_memory=config['initial_memory'], sampling_seed=seed, capture=True)

    def request(method, path, data, timeout):
        options = propose_options(data['query'], data['feedback'], data['memory'])
        # First write a public observation; later recall it. Controls remain forced noop.
        index = next((i for i, o in enumerate(options) if o['action']['op'] == ('write' if seed == 1 else 'recall')), 0)
        return dict(policy_id=identity, query_id=data['query_id'], seed=data['seed'],
            option_index=index, action=options[index]['action'], prompt_token_ids=[1],
            option_token_ids=list(range(len(options))), behavior_logprobs=[-math.log(len(options))] * len(options))
    selected._request = request
    return selected


class NativeRolloutTests(unittest.TestCase):
    def test_clbench_prefix_uses_original_schedule_and_scorer(self):
        from src.interface import ContinualLearningSystem, Response

        class FoldSystem(ContinualLearningSystem):
            name = 'codex'

            def __init__(self, **kwargs):
                pass

            def reset(self):
                pass

            def respond(self, query):
                return Response(query.response_schema(thinking='synthetic executor', action='FOLD'), metadata={})

            def _stop_container(self):
                pass

        config = dict(harness='codex', task='exploitable_poker', split='train',
            task_params=dict(num_instances=4, opponent_policy='calling_station'),
            system_params=dict(backend='skill_rsi', model='synthetic'), instance_indices=[0, 1])
        with TemporaryDirectory() as temporary, patch('skill_rsi_rl.harnesses.codex_system.CodexSystem', FoldSystem):
            result = run_rollout(config, environment_seed=71011, sampling_seed=1, policy_id=None,
                                 output=Path(temporary) / 'prefix', sample=False)
        self.assertEqual(result['expected_instances'], 2)
        self.assertEqual([x['instance_index'] for x in result['outcomes']], [0, 1])
        self.assertEqual(len(result['outcomes']), 2)
        self.assertEqual(result['decisions'], [])

    def run_fixture(self, config, output, seed):
        with patch.object(native_runner, 'prepare', side_effect=lambda x: x), \
                patch('skill_rsi_rl.harnesses.codex_system.CodexSystem', Executor), \
                patch.object(native_runner, 'NativeSession', Session), \
                patch('skill_rsi_rl.harnesses.runner._policy', policy):
            return run_rollout(config, environment_seed=17, sampling_seed=seed, policy_id='frozen',
                               output=output, sample=True)

    def test_dispatch_anchor_control_and_group_validation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for seed in (1, 2):
                Session.reward = float(seed - 1)
                rows.append(self.run_fixture(benchmark(), root / str(seed), seed))
            self.assertEqual(check_group(rows, 'frozen', 2), [0, 1])
            self.assertEqual(rows[0]['bench_commit'], 'synthetic-native-revision')
            self.assertEqual(len(rows[0]['decisions']), 4)
            self.assertEqual(rows[0]['history'][2]['student_before'], rows[0]['history'][0]['student_after'])
            self.assertEqual(rows[0]['decisions'][-1]['state']['memory'], {})
            self.assertEqual(rows[0]['decisions'][-1]['state']['feedback'], [])
            self.assertEqual(Executor.instances[-1].restored, [('episode-0000', True), ('initial-executor', True)])
            self.assertTrue(Executor.instances[-1].closed)
            self.assertTrue((root / '1/trace.json.gz').is_file())
            with self.assertRaises(FileExistsError):
                self.run_fixture(benchmark(), root / '1', 1)

    def test_changed_public_feedback_or_anchor_rejected(self):
        with TemporaryDirectory() as temporary:
            row = self.run_fixture(benchmark(), Path(temporary) / 'run', 1)
            changed = deepcopy(row)
            changed['native_interactions'][0]['public']['tool_results'][0]['content'] = 'hidden answer'
            with self.assertRaisesRegex(ValueError, 'detached'):
                validate_history(changed)
            changed = deepcopy(row)
            changed['history'][2]['student_before']['memory'] = {}
            with self.assertRaisesRegex(ValueError, 'anchor'):
                validate_history(changed)

    def test_native_source_is_rechecked_after_mutation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'source.py').write_text('original')
            import hashlib
            inventory = root / 'source.json'
            inventory.write_text(json.dumps(dict(revision='v1', files={
                'source.py': hashlib.sha256(b'original').hexdigest()})))
            native_spec.verify_source(str(root), str(inventory), 'v1')
            (root / 'source.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'digest differs'):
                native_spec.verify_source(str(root), str(inventory), 'v1')

    def test_codex_baseline_disables_plugins_and_anchor_restores_files(self):
        from pydantic import BaseModel
        from skill_rsi_rl.harnesses.codex_system import CodexSystem
        from src.interface import Query

        class Answer(BaseModel):
            answer: str

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = CodexSystem('synthetic', runtime_root=str(root / 'executor'),
                                 upstream='https://synthetic.test/v1', credential_env='SYNTHETIC_KEY')
            try:
                capture = root / 'capture'
                capture.mkdir()
                result = dict(path=str(capture), status='completed', last_turn_status='completed',
                              final_response='{"answer":"ok"}', session_id='synthetic-session')
                with patch.object(native.adapter, 'launch', return_value=result) as launch, \
                        patch('skill_rsi.evidence.build.build'), patch('skill_rsi.evidence.verify.verify'):
                    native.respond(Query('public query', Answer, instance_id='one', instance_index=0))
                options = launch.call_args.kwargs['options']
                self.assertFalse(options['native_plugin'])
                self.assertFalse(options['baseline_tools'])
                (native.workspace / 'note.txt').write_text('public learned note')
                anchor = root / 'anchor'
                identity = native.snapshot(anchor)
                (native.workspace / 'note.txt').write_text('later branch')
                native.session = 'later-session'
                self.assertEqual(native.restore(anchor, fresh_session=True), identity)
                self.assertEqual((native.workspace / 'note.txt').read_text(), 'public learned note')
                self.assertIsNone(native.session)
                self.assertFalse((anchor / 'home/auth.json').exists())
                (anchor / 'workspace/note.txt').write_text('tampered')
                with self.assertRaisesRegex(ValueError, 'changed after snapshot'):
                    native.restore(anchor, fresh_session=True)
            finally:
                native.close()

    def test_api_source_split_and_episode_plan_are_pinned(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = root / 'task'
            (task / 'environment').mkdir(parents=True)
            (task / 'instruction.md').write_text('public task')
            (task / 'task.toml').write_text('[task]\nname="example"\n[environment]\nmcp_servers=[]\n')
            (root / 'split.json').write_text(json.dumps(dict(train=['example'], dev=['other'], holdout=['held'])))
            (root / 'inventory.json').write_text(json.dumps(dict(revision='v1', files={'task/task.toml': native_spec.file_sha(task / 'task.toml')})))
            config = dict(kind='workbuddy', harness='codex', split='train', system_params={'backend': 'skill_rsi'},
                native=dict(repository=str(root), source_manifest=str(root / 'inventory.json'), revision='v1',
                            task_dir='task', python=sys.executable, split_manifest=str(root / 'split.json')))
            pinned = native_spec.prepare(config)
            self.assertEqual(pinned['task'], 'workbuddy')
            self.assertEqual(pinned['native']['lineage_id'], 'example')
            self.assertEqual(native_spec.benchmark_commit(pinned), 'v1')
            changed = deepcopy(pinned)
            changed['split'] = 'dev'
            with self.assertRaisesRegex(ValueError, 'assigned'):
                native_spec.prepare(changed)
            (root / 'split.json').write_text(json.dumps(dict(train=['example'], dev=['example'], holdout=['held'])))
            with self.assertRaisesRegex(ValueError, 'cross'):
                native_spec.prepare(pinned)


if __name__ == '__main__':
    unittest.main()
