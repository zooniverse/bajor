import json

from zoobot.shared import schemas as zoobot_schemas
from zoobot.shared.schemas import Schema


SCHEMA_ALIASES = {
    'cosmic_dawn': 'cosmic_dawn_ortho_schema',
    'euclid': 'euclid_ortho_schema',
    'jwst_cosmos': 'gz_jwst_schema',
}


def load_schema(schema_name, custom_schema_json=None):
    schema_name = normalize_schema_name(schema_name)
    custom_schema = parse_custom_schema_json(custom_schema_json)

    if schema_name:
        schema = load_zoobot_schema(schema_name)
        if schema is not None:
            return schema

        return load_custom_schema_or_raise(custom_schema, schema_name)

    if custom_schema:
        return build_custom_schema(custom_schema)

    return load_required_zoobot_schema('cosmic_dawn')


def normalize_schema_name(schema_name):
    if schema_name is not None:
        return schema_name.strip()

    return None


def parse_custom_schema_json(custom_schema_json):
    if custom_schema_json:
        return json.loads(custom_schema_json)

    return None


def load_custom_schema_or_raise(custom_schema, schema_name):
    if custom_schema:
        return build_custom_schema(custom_schema)

    raise ValueError(f'Unknown schema: {schema_name}')


def load_required_zoobot_schema(schema_name):
    schema = load_zoobot_schema(schema_name)
    if schema is not None:
        return schema

    raise ValueError(f'Unknown schema: {schema_name}')


def load_zoobot_schema(schema_name):
    for attribute_name in zoobot_schema_attribute_names(schema_name):
        schema = getattr(zoobot_schemas, attribute_name, None)
        if schema is not None:
            return schema

    return None


def build_custom_schema(custom_schema):
    pairs = custom_schema.get('question_answer_pairs')

    if pairs is None:
        raise ValueError('Custom schema must include question_answer_pairs')

    dependencies = custom_schema.get('dependencies') or {}
    dependencies = {question: dependencies.get(question) for question in pairs}

    return Schema(pairs, dependencies)



def zoobot_schema_attribute_names(schema_name):
    if schema_name in SCHEMA_ALIASES:
        yield SCHEMA_ALIASES[schema_name]

    yield schema_name

    if not schema_name.endswith('_schema'):
        yield f'{schema_name}_schema'
