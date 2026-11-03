"""Conservative source-only release check; never reads the research dataset."""
import argparse
import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ROOT_FILES={'.gitignore','.gitattributes','README.md','requirements.txt'}
ALLOWED={'scripts_qc':{'.py'},'tools':{'.py'},'tests':{'.py'},'docs':{'.md'},'configs':{'.yaml'}}
SECRET_PATTERNS=[r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{30,}',r'AKIA[0-9A-Z]{16}',r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',r'https?://[^\s/@:]+:[^\s/@]+@',r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.(?:com|cn|net)\b']

def files(staged=False):
    if staged:
        names=subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=ACMR','-z'],cwd=ROOT).decode().split('\0')
        for name in filter(None,names):
            yield Path(name),subprocess.check_output(['git','show',':'+name],cwd=ROOT)
    else:
        for p in ROOT.rglob('*'):
            rel=p.relative_to(ROOT)
            if p.is_file() and not any(s in {'.git','__pycache__','.pytest_cache'} for s in rel.parts):
                yield rel,p.read_bytes()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--staged',action='store_true');a=ap.parse_args()
    errors=[];count=0;parsed=0
    for rel,raw in files(a.staged):
        name=rel.as_posix();count+=1
        allowed=name in ROOT_FILES or (rel.parts[0] in ALLOWED and rel.suffix in ALLOWED[rel.parts[0]]) or name in {'docs/source_manifest.json','docs/baseline_source_manifest.json','docs/phpr_configuration_audit.json','docs/upstream_source_manifest.json'} or (name.startswith('experiments/aggregation_comparison/') and rel.suffix=='.py')
        if not allowed:errors.append(name+': file is outside source-only whitelist')
        if b'\0' in raw:errors.append(name+': binary content');continue
        try:text=raw.decode('utf-8-sig')
        except UnicodeDecodeError:errors.append(name+': not UTF-8');continue
        for pattern in SECRET_PATTERNS:
            if re.search(pattern,text,re.I):errors.append(name+': possible credential or private email (content withheld)')
        # Check Python string literals, not regex source code, for personal paths.
        if rel.suffix=='.py':
            tree=ast.parse(text,filename=name);parsed+=1
            for node in ast.walk(tree):
                if isinstance(node,ast.Constant) and isinstance(node.value,str):
                    value=node.value.strip()
                    if re.match(r'^(?:/[hH]ome/|/[uU]sers/|[A-Za-z]:[\\/])',value):errors.append(name+': absolute local path literal')
        if re.search(r'(?:/[hH]ome/[\w.-]+/|[A-Za-z]:[\\/]Users[\\/])',text):errors.append(name+': personal absolute path')
    imported=[]
    if not a.staged and not errors:
        for path in sorted((ROOT/'scripts_qc').glob('*.py')):
            spec=importlib.util.spec_from_file_location('_release_'+path.stem,path)
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            imported.append(path.name)
    print(json.dumps({'passed':not errors,'staged':a.staged,'reviewed_files':count,'parsed_python_files':parsed,'imported_modules':imported,'errors':errors},indent=2))
    raise SystemExit(bool(errors))

if __name__=='__main__':main()
