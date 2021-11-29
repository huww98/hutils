from typing import Dict
from _jsonnet import evaluate_snippet
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def config_snippet(ext_config_count: int):
    snippet = 'local base = import "__base_config__";\n'
    for i in range(ext_config_count):
        snippet += f'local arg{i} = import "__arg_{i}__";\n'

    snippet += 'base'
    for i in range(ext_config_count):
        snippet += f' + arg{i}'
    return snippet


def ext_config_template(ext_config: str):
    return f'local add = import "__addition_config__";\n{ext_config}'


def try_path(dir, rel):
    if rel[0] == '/':
        full_path = rel
    else:
        full_path = dir + rel

    with open(full_path) as f:
        return full_path, f.read()


arg_regex = re.compile('^__arg_(\d+)__$')


def get_config(args) -> Dict:
    def import_callback(dir, rel):
        arg_match = arg_regex.match(rel)
        if arg_match is not None:
            full_path = rel
            index = int(arg_match.group(1))
            content = ext_config_template(args.ext_config[index])
        else:
            if rel == '__base_config__':
                rel = Path(args.config)
            elif rel == '__addition_config__':
                rel = Path(args.config).with_name('addition.libsonnet')
            else:
                rel = Path(rel)
            full_path = dir / rel
            full_path = str(full_path)
            with open(full_path) as f:
                content = f.read()
        return full_path, content

    json_str = evaluate_snippet(
        '__composed_config__',
        config_snippet(len(args.ext_config)),
        import_callback=import_callback,
    )

    json_obj = json.loads(json_str)

    return json_obj
