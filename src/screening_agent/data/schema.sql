PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_number TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    clinical_context TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_patients_full_name ON patients(full_name);
CREATE INDEX IF NOT EXISTS idx_patients_security_number ON patients(security_number);
