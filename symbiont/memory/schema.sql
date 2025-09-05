CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, started_at INTEGER, ended_at INTEGER, status TEXT);
CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, created_at INTEGER, tags TEXT);
CREATE TABLE IF NOT EXISTS beliefs (id INTEGER PRIMARY KEY AUTOINCREMENT, statement TEXT, confidence REAL DEFAULT 0.5, evidence_json TEXT, created_at INTEGER DEFAULT (strftime('%s','now')), updated_at INTEGER DEFAULT (strftime('%s','now')));
CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER, description TEXT, status TEXT, assignee_role TEXT, result TEXT, created_at INTEGER, updated_at INTEGER);
CREATE TABLE IF NOT EXISTS artifacts (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER, type TEXT, path TEXT, summary TEXT, created_at INTEGER);
CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, ref_table TEXT, ref_id INTEGER, embedding TEXT);
