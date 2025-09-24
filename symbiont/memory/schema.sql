CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, started_at INTEGER, ended_at INTEGER, status TEXT);
CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, created_at INTEGER, tags TEXT);
CREATE TABLE IF NOT EXISTS beliefs (id INTEGER PRIMARY KEY AUTOINCREMENT, statement TEXT, confidence REAL DEFAULT 0.5, evidence_json TEXT, created_at INTEGER DEFAULT (strftime('%s','now')), updated_at INTEGER DEFAULT (strftime('%s','now')));
CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER, description TEXT, status TEXT, assignee_role TEXT, result TEXT, created_at INTEGER, updated_at INTEGER);
CREATE TABLE IF NOT EXISTS artifacts (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER, type TEXT, path TEXT, summary TEXT, created_at INTEGER);
CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, ref_table TEXT, ref_id INTEGER, embedding TEXT);
CREATE TABLE IF NOT EXISTS audits (id INTEGER PRIMARY KEY AUTOINCREMENT, capability TEXT, description TEXT, preview TEXT, approved INTEGER, created_at INTEGER DEFAULT (strftime('%s','now')));
CREATE TABLE IF NOT EXISTS intents (id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER, summary TEXT, created_at INTEGER DEFAULT (strftime('%s','now')), updated_at INTEGER DEFAULT (strftime('%s','now')));
-- GraphRAG-lite tables
CREATE TABLE IF NOT EXISTS entities (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS relations (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS claims (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  subject_id INTEGER,
  relation_id INTEGER,
  object TEXT,
  importance REAL DEFAULT 0.5,
  source_url TEXT,
  created_at INTEGER DEFAULT (strftime('%s','now')),
  updated_at INTEGER DEFAULT (strftime('%s','now'))
);
-- Link artifacts (e.g., notes) to episodes for "Sources used this cycle"
CREATE TABLE IF NOT EXISTS episode_artifacts (
  episode_id INTEGER,
  artifact_id INTEGER,
  linked_at INTEGER DEFAULT (strftime('%s','now'))
);
