-- Legal Assistant AI Database Schema (Phase 2)
-- Normalized schema for Persian legal documents
-- SQLite with FTS5 full-text search support

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- ========================================
-- MAIN TABLES
-- ========================================

-- Documents table: Main document metadata
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    document_uid TEXT NOT NULL UNIQUE,
    title TEXT,
    document_type TEXT,
    approval_date TEXT,
    approval_authority TEXT,
    section_name TEXT,
    confidence_score REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) WITHOUT ROWID;

-- Chapters table: Document chapters/sections
CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    chapter_title TEXT,
    chapter_index INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Articles table: Individual articles within chapters
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id INTEGER NOT NULL,
    article_number TEXT,
    article_text TEXT,
    FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE
);

-- Notes table: تبصره (notes/remarks) for articles
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL,
    note_label TEXT,
    note_text TEXT,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

-- Clauses table: بند (clauses) within notes
CREATE TABLE IF NOT EXISTS clauses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL,
    clause_label TEXT,
    clause_text TEXT,
    FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
);

-- ========================================
-- INDEXES FOR PERFORMANCE
-- ========================================

-- Foreign key indexes
CREATE INDEX IF NOT EXISTS idx_chapters_document_id ON chapters(document_id);
CREATE INDEX IF NOT EXISTS idx_articles_chapter_id ON articles(chapter_id);
CREATE INDEX IF NOT EXISTS idx_notes_article_id ON notes(article_id);
CREATE INDEX IF NOT EXISTS idx_clauses_note_id ON clauses(note_id);

-- Frequently queried columns
CREATE INDEX IF NOT EXISTS idx_documents_document_type ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_section_name ON documents(section_name);
CREATE INDEX IF NOT EXISTS idx_documents_approval_authority ON documents(approval_authority);
CREATE INDEX IF NOT EXISTS idx_documents_document_uid ON documents(document_uid);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_articles_article_number ON articles(article_number);

-- ========================================
-- FULL-TEXT SEARCH TABLES (FTS5)
-- ========================================

-- Documents FTS table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title,
    document_type,
    approval_authority,
    section_name,
    content='documents',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 1 tokenchars ''ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی‌آأإٱٲٳٴٵٶٷٸٹٺٻټٽپژچکگڈڑںۂۃ'''
);

-- Articles FTS table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
    article_number,
    article_text,
    content='articles',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 1 tokenchars ''ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی‌آأإٱٲٳٴٵٶٷٸٹٺٻټٽپژچکگڈڑںۂۃ'''
);

-- Notes FTS table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    note_label,
    note_text,
    content='notes',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 1 tokenchars ''ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی‌آأإٱٲٳٴٵٶٷٸٹٺٻټٽپژچکگڈڑںۂۃ'''
);

-- Clauses FTS table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS clauses_fts USING fts5(
    clause_label,
    clause_text,
    content='clauses',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 1 tokenchars ''ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی‌آأإٱٲٳٴٵٶٷٸٹٺٻټٽپژچکگڈڑںۂۃ'''
);

-- ========================================
-- TRIGGERS FOR FTS SYNCHRONIZATION
-- ========================================

-- Documents triggers
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, document_type, approval_authority, section_name) 
    VALUES (new.id, new.title, new.document_type, new.approval_authority, new.section_name);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, document_type, approval_authority, section_name) 
    VALUES ('delete', old.id, old.title, old.document_type, old.approval_authority, old.section_name);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, document_type, approval_authority, section_name) 
    VALUES ('delete', old.id, old.title, old.document_type, old.approval_authority, old.section_name);
    INSERT INTO documents_fts(rowid, title, document_type, approval_authority, section_name) 
    VALUES (new.id, new.title, new.document_type, new.approval_authority, new.section_name);
END;

-- Articles triggers
CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles BEGIN
    INSERT INTO articles_fts(rowid, article_number, article_text) 
    VALUES (new.id, new.article_number, new.article_text);
END;

CREATE TRIGGER IF NOT EXISTS articles_ad AFTER DELETE ON articles BEGIN
    INSERT INTO articles_fts(articles_fts, rowid, article_number, article_text) 
    VALUES ('delete', old.id, old.article_number, old.article_text);
END;

CREATE TRIGGER IF NOT EXISTS articles_au AFTER UPDATE ON articles BEGIN
    INSERT INTO articles_fts(articles_fts, rowid, article_number, article_text) 
    VALUES ('delete', old.id, old.article_number, old.article_text);
    INSERT INTO articles_fts(rowid, article_number, article_text) 
    VALUES (new.id, new.article_number, new.article_text);
END;

-- Notes triggers
CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, note_label, note_text) 
    VALUES (new.id, new.note_label, new.note_text);
END;

CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, note_label, note_text) 
    VALUES ('delete', old.id, old.note_label, old.note_text);
END;

CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, note_label, note_text) 
    VALUES ('delete', old.id, old.note_label, old.note_text);
    INSERT INTO notes_fts(rowid, note_label, note_text) 
    VALUES (new.id, new.note_label, new.note_text);
END;

-- Clauses triggers
CREATE TRIGGER IF NOT EXISTS clauses_ai AFTER INSERT ON clauses BEGIN
    INSERT INTO clauses_fts(rowid, clause_label, clause_text) 
    VALUES (new.id, new.clause_label, new.clause_text);
END;

CREATE TRIGGER IF NOT EXISTS clauses_ad AFTER DELETE ON clauses BEGIN
    INSERT INTO clauses_fts(clauses_fts, rowid, clause_label, clause_text) 
    VALUES ('delete', old.id, old.clause_label, old.clause_text);
END;

CREATE TRIGGER IF NOT EXISTS clauses_au AFTER UPDATE ON clauses BEGIN
    INSERT INTO clauses_fts(clauses_fts, rowid, clause_label, clause_text) 
    VALUES ('delete', old.id, old.clause_label, old.clause_text);
    INSERT INTO clauses_fts(rowid, clause_label, clause_text) 
    VALUES (new.id, new.clause_label, new.clause_text);
END;

-- ========================================
-- UPDATE TIMESTAMP TRIGGER
-- ========================================

-- Automatically update the updated_at timestamp when documents are modified
CREATE TRIGGER IF NOT EXISTS update_documents_timestamp 
AFTER UPDATE ON documents
BEGIN
    UPDATE documents SET updated_at = datetime('now') WHERE id = NEW.id;
END;

-- ========================================
-- UTILITY VIEWS
-- ========================================

-- View to get complete document hierarchy
CREATE VIEW IF NOT EXISTS document_hierarchy AS
SELECT 
    d.id as document_id,
    d.document_uid,
    d.title as document_title,
    d.document_type,
    d.section_name,
    c.id as chapter_id,
    c.chapter_index,
    c.chapter_title,
    a.id as article_id,
    a.article_number,
    a.article_text,
    n.id as note_id,
    n.note_label,
    n.note_text,
    cl.id as clause_id,
    cl.clause_label,
    cl.clause_text
FROM documents d
LEFT JOIN chapters c ON d.id = c.document_id
LEFT JOIN articles a ON c.id = a.chapter_id
LEFT JOIN notes n ON a.id = n.article_id
LEFT JOIN clauses cl ON n.id = cl.note_id;

-- ========================================
-- INITIALIZATION COMPLETE
-- ========================================

-- Rebuild FTS indexes to ensure consistency
INSERT OR IGNORE INTO documents_fts(documents_fts) VALUES('rebuild');
INSERT OR IGNORE INTO articles_fts(articles_fts) VALUES('rebuild');
INSERT OR IGNORE INTO notes_fts(notes_fts) VALUES('rebuild');
INSERT OR IGNORE INTO clauses_fts(clauses_fts) VALUES('rebuild');