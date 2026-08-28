import json
import pathlib
import sys
import types


scripts_path = pathlib.Path(__file__).resolve().parents[1] / 'azure' / 'batch' / 'scripts'
sys.path.insert(0, str(scripts_path))


class FakeSchema:
    def __init__(self, question_answer_pairs, dependencies):
        self.question_answer_pairs = question_answer_pairs
        self.dependencies = dependencies
        self.label_cols = [
            f'{question}{answer}'
            for question, answers in question_answer_pairs.items()
            for answer in answers
        ]


fake_zoobot = types.ModuleType('zoobot')
fake_zoobot_shared = types.ModuleType('zoobot.shared')
fake_zoobot_schemas = types.SimpleNamespace()
fake_zoobot_schema_module = types.ModuleType('zoobot.shared.schemas')
fake_zoobot_schema_module.Schema = FakeSchema
fake_zoobot_shared.schemas = fake_zoobot_schemas
sys.modules.setdefault('zoobot', fake_zoobot)
sys.modules.setdefault('zoobot.shared', fake_zoobot_shared)
sys.modules.setdefault('zoobot.shared.schemas', fake_zoobot_schema_module)

from schema_resolver import build_custom_schema, load_schema


def test_build_custom_schema_without_dependencies_uses_root_questions():
    schema_payload = {
        'question_answer_pairs': {
            'merger': ['_yes', '_no', '_artifact'],
        },
    }

    schema = build_custom_schema(schema_payload)

    assert schema.question_answer_pairs == {
        'merger': ['_yes', '_no', '_artifact'],
    }
    assert schema.dependencies == {'merger': None}
    assert schema.label_cols == ['merger_yes', 'merger_no', 'merger_artifact']


def test_load_schema_uses_custom_schema_json_for_unknown_zoobot_schema():
    schema_payload = {
        'question_answer_pairs': {
            'merger': ['_yes', '_no', '_artifact'],
        },
        'dependencies': {
            'merger': None,
        },
    }

    schema = load_schema('gztt', custom_schema_json=json.dumps(schema_payload))

    assert schema.label_cols == ['merger_yes', 'merger_no', 'merger_artifact']


def test_load_schema_uses_custom_schema_json_when_schema_name_is_missing():
    schema_payload = {
        'question_answer_pairs': {
            'merger': ['_yes', '_no', '_artifact'],
        },
        'dependencies': {
            'merger': None,
        },
    }

    schema = load_schema(None, custom_schema_json=json.dumps(schema_payload))

    assert schema.label_cols == ['merger_yes', 'merger_no', 'merger_artifact']


def test_load_schema_uses_default_schema_when_schema_name_and_custom_schema_are_missing():
    fake_zoobot_schemas.cosmic_dawn_ortho_schema = FakeSchema(
        {'smooth-or-featured': ['_smooth', '_featured']},
        {'smooth-or-featured': None},
    )

    schema = load_schema(None)

    assert schema.label_cols == ['smooth-or-featured_smooth', 'smooth-or-featured_featured']


def test_load_schema_raises_for_unknown_schema_without_custom_schema():
    try:
        load_schema('missing_schema')
    except ValueError as error:
        assert str(error) == 'Unknown schema: missing_schema'
    else:
        raise AssertionError('Expected unknown schema to raise')
