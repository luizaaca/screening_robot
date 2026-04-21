PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_number TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    birth_date TEXT,
    age_years INTEGER,
    sex TEXT,
    allergies_json TEXT NOT NULL DEFAULT '[]',
    conditions_json TEXT NOT NULL DEFAULT '[]',
    medications_json TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_patients_full_name ON patients(full_name);
CREATE INDEX IF NOT EXISTS idx_patients_security_number ON patients(security_number);

CREATE TABLE IF NOT EXISTS patient_vitals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    recorded_at TEXT NOT NULL,
    blood_pressure TEXT,
    heart_rate_bpm INTEGER,
    respiratory_rate_bpm INTEGER,
    temperature_c REAL,
    oxygen_saturation_pct INTEGER,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_patient_vitals_patient_id ON patient_vitals(patient_id, recorded_at DESC);

CREATE TABLE IF NOT EXISTS patient_exams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    exam_name TEXT NOT NULL,
    exam_date TEXT,
    result_summary TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_patient_exams_patient_id ON patient_exams(patient_id, exam_date DESC);
