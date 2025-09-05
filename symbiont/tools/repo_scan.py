from __future__ import annotations
import os
def _has(path,names): return any(os.path.exists(os.path.join(path,n)) for n in names)
def _present(path,name): return os.path.exists(os.path.join(path,name))
def inspect_repo(path: str = '.'):
    path=os.path.abspath(path)
    try: files=set(os.listdir(path))
    except Exception: files=set()
    d={'path':path,'git':_present(path,'.git'),'node':_has(path,['package.json','pnpm-lock.yaml','yarn.lock']),
       'python':_has(path,['pyproject.toml','requirements.txt','.python-version']),'docker':_has(path,['Dockerfile','docker-compose.yml','docker-compose.yaml']),
       'ci':_present(path,'.github') or _present(path,'.gitlab-ci.yml'),'precommit':_has(path,['.pre-commit-config.yaml','.pre-commit-config.yml']),
       'editorconfig':_present(path,'.editorconfig'),'readme':_has(path,['README.md','readme.md']),'license':_has(path,['LICENSE','LICENSE.md']),
       'env_files':any(f.startswith('.env') for f in files),'tests':any(d in files for d in ['tests','test','__tests__'])}
    s=[]
    def add(t,det,mins,prio=5,cmds=None): s.append({'title':t,'details':det,'effort_min':mins,'priority':prio,'commands':cmds or []})
    if not d['editorconfig']: add('.editorconfig for consistent formatting','Add a minimal .editorconfig.',3,1,["printf '*\nend_of_line = lf\ninsert_final_newline = true\nindent_style = space\nindent_size = 2\n' > .editorconfig"])
    if not d['precommit']: add('Pre-commit hooks','Install pre-commit with simple hook.',8,2,["pip install pre-commit && printf 'repos: []\n' > .pre-commit-config.yaml && pre-commit install"])
    if not d['readme']: add('Add README.md skeleton','Create top-level README.',6,3,["printf '# Project\n\nSetup:\n\nRun:\n' > README.md"])
    if not d['license']: add('Add LICENSE','Clarify usage.',2,5,["printf 'MIT\n' > LICENSE"])
    if d['python'] and not _present(path,'ruff.toml'): add('Add ruff (linter/formatter)','One tool to lint + format.',8,2,["pip install ruff && printf '[format]\nline-length = 100\n' > ruff.toml"])
    if d['node'] and not _present(path,'.nvmrc'): add('Add .nvmrc','Pin Node version.',2,2,["node -v | sed 's/v//' > .nvmrc || echo '20' > .nvmrc"])
    if d['node'] and not any(n in files for n in ['.eslintrc','.eslintrc.json','.eslintrc.js']): add('Quick ESLint init','Bootstrap ESLint.',10,3,["npx eslint --init || echo '{\"extends\":[\"eslint:recommended\"]}' > .eslintrc.json"])
    if d['node'] and not any(n in files for n in ['.prettierrc','.prettierrc.json']): add('Add Prettier config','Consistent formatting.',2,2,["printf '{\"printWidth\":100,\"semi\":true}' > .prettierrc"])
    if d['docker'] and not _present(path,'.dockerignore'): add('Add .dockerignore','Shrink build context.',3,2,["printf '.git\nnode_modules\n.vscode\n__pycache__\n' > .dockerignore"])
    if not d['ci'] and d['git']: add('Add minimal CI workflow','Run tests/lints on PR.',10,4,["mkdir -p .github/workflows && printf 'name: ci\non: [push]\n' > .github/workflows/ci.yml"])
    if not d['tests']: add('Create tests/ placeholder','Add a trivial test.',3,5,["mkdir -p tests && printf 'def test_placeholder():\n    assert True\n' > tests/test_placeholder.py"])
    s.sort(key=lambda x:(x['effort_min'],x['priority'],x['title'])); return {'detected':d,'suggestions':s}
