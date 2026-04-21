from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

INPUT_CSV = Path("combined_diseases_symptoms_2.csv")
OUTPUT_CSV = Path("combined_diseases_symptoms_2_enriched_with_exams_v2.csv")
PROGRESS_JSON = Path("combined_diseases_symptoms_2_enriched_with_exams_v2.progress.json")
RESEARCH_CACHE_JSON = Path("combined_diseases_symptoms_2_enriched_with_exams_v2.research_cache.json")
BATCH_SIZE = 10
CACHE_SAVE_INTERVAL_SECONDS = 90.0
MAX_IN_MEMORY_TOPIC_PAGES = 256
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DISCLAIMER = (
    "Disclaimer: This is an AI auxiliary tool designed for healthcare professionals. "
    "It is not 100% precise and does not replace a professional medical diagnosis."
)
OUTPUT_FIELDS = [
    "input",
    "output",
    "source",
    "support_status",
    "candidate_diseases",
    "recommended_exams_tests",
    "reasoning",
]
GENERIC_DISEASE_TITLES = {
    "dyspnea",
    "abdominal pain",
    "chest pain",
    "syncope",
    "dizziness",
    "palpitations",
    "nausea and vomiting",
    "cough",
    "fever",
    "headache",
    "back pain",
    "fatigue",
    "amenorrhea",
    "vaginal bleeding",
    "vaginal discharge",
    "rash",
    "joint pain",
    "pain",
    "shortness of breath",
    "breathing difficulty",
}
GENERIC_SYMPTOM_WORDS = {
    "pain",
    "symptoms",
    "problem",
    "problems",
    "abnormal",
    "appearing",
    "feeling",
    "feels",
    "other",
}
TEST_CATALOG = [
    "electrocardiogram (ECG)",
    "Holter monitoring",
    "echocardiogram",
    "pulse oximetry",
    "chest X-ray",
    "chest CT",
    "thyroid function tests",
    "complete blood count (CBC)",
    "basic metabolic panel (BMP)",
    "comprehensive metabolic panel (CMP)",
    "arterial blood gas",
    "spirometry",
    "pregnancy test",
    "urinalysis",
    "urine culture",
    "renal and bladder ultrasound",
    "pelvic examination",
    "pelvic ultrasonography",
    "abdominal ultrasound",
    "CT abdomen and pelvis",
    "serum lipase",
    "liver function tests",
    "rapid strep test",
    "throat culture",
    "laryngoscopy",
    "ENT examination",
    "visual acuity testing",
    "slit-lamp examination",
    "fluorescein staining",
    "ophthalmologic examination",
    "skin scraping or fungal culture",
    "skin swab or bacterial culture",
    "skin biopsy",
    "dermatologic examination",
    "X-ray",
    "joint X-ray",
    "MRI of the affected region",
    "neurologic examination",
    "psychiatric evaluation",
    "toxicology screening",
    "serum glucose",
]


@dataclass(frozen=True)
class DiseaseMetrics:
    score: float
    precision: float
    recall: float
    matched_symptoms: tuple[str, ...]
    matched_weight: float


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slugify_disease(text: str) -> str:
    text = normalize_spaces(text.lower())
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = text.replace("—", " ").replace("–", " ")
    text = re.sub(r"[^a-z0-9/ +'-]", " ", text)
    return normalize_spaces(text)


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def unique_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        cleaned = normalize_spaces(item)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(cleaned)
    return ordered


def clamp_tests(tests: Iterable[str], limit: int = 5) -> list[str]:
    chosen = unique_preserve(tests)
    return chosen[:limit] if chosen else ["complete blood count (CBC)"]


class SymptomModel:
    def __init__(self) -> None:
        self.disease_row_counts: Counter[str] = Counter()
        self.disease_symptom_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.symptom_document_counts: Counter[str] = Counter()
        self.symptom_idf: dict[str, float] = {}
        self.symptom_to_disease_weight: dict[str, dict[str, float]] = defaultdict(dict)
        self.disease_top_weights: dict[str, dict[str, float]] = {}
        self.disease_top_totals: dict[str, float] = {}
        self.total_diseases = 0

    @staticmethod
    def parse_symptoms(raw: str) -> list[str]:
        return [normalize_spaces(part.lower()) for part in raw.split(",") if part.strip()]

    def build(self, input_csv: Path) -> None:
        with input_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                disease = slugify_disease(row["output"])
                symptoms = self.parse_symptoms(row["input"])
                if not disease or not symptoms:
                    continue
                self.disease_row_counts[disease] += 1
                for symptom in symptoms:
                    self.disease_symptom_counts[disease][symptom] += 1

        for symptom_counts in self.disease_symptom_counts.values():
            for symptom in symptom_counts:
                self.symptom_document_counts[symptom] += 1

        self.total_diseases = len(self.disease_row_counts)
        for symptom, count in self.symptom_document_counts.items():
            self.symptom_idf[symptom] = math.log((1 + self.total_diseases) / (1 + count)) + 1.0

        for disease, symptom_counts in self.disease_symptom_counts.items():
            disease_total = self.disease_row_counts[disease]
            weighted: dict[str, float] = {}
            for symptom, count in symptom_counts.items():
                prevalence = count / disease_total
                weight = prevalence * self.symptom_idf.get(symptom, 1.0)
                weighted[symptom] = weight
                self.symptom_to_disease_weight[symptom][disease] = weight

            top_items = sorted(weighted.items(), key=lambda item: item[1], reverse=True)[:12]
            self.disease_top_weights[disease] = dict(top_items)
            self.disease_top_totals[disease] = sum(weight for _, weight in top_items) or 1.0

    def top_symptoms(self, disease: str, limit: int = 6) -> list[str]:
        return list(self.disease_top_weights.get(disease, {}).keys())[:limit]

    def score_row(self, symptoms: list[str]) -> dict[str, DiseaseMetrics]:
        accum: dict[str, float] = defaultdict(float)
        matched_by_disease: dict[str, set[str]] = defaultdict(set)
        unique_symptoms = sorted(set(symptoms))
        row_weight_total = sum(self.symptom_idf.get(symptom, 1.0) for symptom in unique_symptoms) or 1.0

        for symptom in unique_symptoms:
            for disease, weight in self.symptom_to_disease_weight.get(symptom, {}).items():
                accum[disease] += weight
                matched_by_disease[disease].add(symptom)

        scores: dict[str, DiseaseMetrics] = {}
        for disease, matched_weight in accum.items():
            precision = min(1.0, matched_weight / row_weight_total)
            recall = min(1.0, matched_weight / self.disease_top_totals.get(disease, 1.0))
            score = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
            scores[disease] = DiseaseMetrics(
                score=score,
                precision=precision,
                recall=recall,
                matched_symptoms=tuple(sorted(matched_by_disease[disease])),
                matched_weight=matched_weight,
            )
        return scores


class WebResearcher:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.cache = self._load_cache()
        self.cache_dirty = False
        self.last_cache_save = time.monotonic()

    def _load_cache(self) -> dict:
        empty_cache = {"symptom_queries": {}, "disease_queries": {}, "topic_pages": {}}
        if not self.cache_path.exists():
            return empty_cache
        try:
            raw_cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return empty_cache

        symptom_queries = raw_cache.get("symptom_queries", {})
        disease_queries = raw_cache.get("disease_queries", {})
        return {
            "symptom_queries": symptom_queries if isinstance(symptom_queries, dict) else {},
            "disease_queries": disease_queries if isinstance(disease_queries, dict) else {},
            "topic_pages": {},
        }

    def _persistent_cache_payload(self) -> dict[str, dict]:
        return {
            "symptom_queries": self.cache.get("symptom_queries", {}),
            "disease_queries": self.cache.get("disease_queries", {}),
        }

    def _trim_topic_pages(self) -> None:
        topic_pages = self.cache.setdefault("topic_pages", {})
        while len(topic_pages) > MAX_IN_MEMORY_TOPIC_PAGES:
            oldest_key = next(iter(topic_pages))
            del topic_pages[oldest_key]

    def mark_cache_dirty(self) -> None:
        self.cache_dirty = True

    def save_cache(self) -> None:
        if not self.cache_dirty:
            return
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        payload = self._persistent_cache_payload()
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tmp_path.replace(self.cache_path)
        self.cache_dirty = False
        self.last_cache_save = time.monotonic()

    def maybe_save_cache(self, force: bool = False) -> None:
        if not self.cache_dirty:
            return
        if force or (time.monotonic() - self.last_cache_save) >= CACHE_SAVE_INTERVAL_SECONDS:
            self.save_cache()

    def get(self, url: str) -> str:
        if url in self.cache["topic_pages"]:
            return self.cache["topic_pages"][url]
        response = self.session.get(url, timeout=20)
        response.raise_for_status()
        text = response.text
        self.cache["topic_pages"][url] = text
        self._trim_topic_pages()
        time.sleep(0.05)
        return text

    def merck_search(self, query: str) -> list[dict[str, str]]:
        query_key = normalize_spaces(query.lower())
        cached = self.cache["symptom_queries"].get(query_key)
        if cached:
            return cached["results"]

        url = f"https://www.nhs.uk/search/results?q={quote_plus(query)}"
        html = self.get(url)
        soup = BeautifulSoup(html, "html.parser")
        results: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for anchor in soup.select('a[href*="/search/click?url="]'):
            href = anchor.get("href")
            text = normalize_spaces(anchor.get_text(" ", strip=True))
            if not isinstance(href, str):
                continue
            parsed = urlparse(href)
            raw_target = parse_qs(parsed.query).get("url", [""])[0]
            resolved = urljoin("https://www.nhs.uk", raw_target)
            lowered = resolved.lower()
            title = slugify_disease(text)
            if not title or len(title) < 3:
                continue
            if any(fragment in lowered for fragment in (
                "/medicines/",
                "/tests-and-treatments/",
                "/our-policies/",
                "/service-search/",
                "/nhs-app/",
                "/mental-health/self-help/",
                "/treatment/",
            )):
                continue
            if resolved in seen_urls:
                continue
            seen_urls.add(resolved)
            results.append({"title": text, "url": resolved})
            if len(results) >= 8:
                break

        self.cache["symptom_queries"][query_key] = {"query": query, "results": results}
        self.mark_cache_dirty()
        return results

    def disease_page(self, disease: str) -> dict[str, object]:
        disease_key = slugify_disease(disease)
        cached = self.cache["disease_queries"].get(disease_key)
        if cached:
            return cached

        results = self.merck_search(disease_key)
        chosen = next(
            (
                result
                for result in results
                if disease_key in slugify_disease(result["title"]) or slugify_disease(result["title"]) in disease_key
            ),
            results[0] if results else {"title": disease_key, "url": ""},
        )

        page_text = ""
        extracted_tests: list[str] = []
        if chosen.get("url"):
            try:
                html = self.get(chosen["url"])
                page_text = self.extract_main_text(html)
                extracted_tests = self.extract_tests_from_text(page_text)
            except requests.RequestException:
                page_text = ""

        payload = {
            "disease": disease_key,
            "search_title": chosen.get("title", disease_key),
            "url": chosen.get("url", ""),
            "tests": extracted_tests,
            "text_excerpt": page_text[:5000],
        }
        self.cache["disease_queries"][disease_key] = payload
        self.mark_cache_dirty()
        return payload

    @staticmethod
    def extract_main_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = normalize_spaces(soup.get_text("\n", strip=True))
        return text

    @staticmethod
    def extract_tests_from_text(text: str) -> list[str]:
        lowered = text.lower()
        found: list[str] = []
        for test_name in TEST_CATALOG:
            if test_name.lower() in lowered:
                found.append(test_name)
        return clamp_tests(found, limit=5)


def keyword_score(symptom: str, model: SymptomModel) -> tuple[float, int, int]:
    words = [word for word in re.split(r"[^a-z0-9]+", symptom.lower()) if word]
    generic_penalty = sum(word in GENERIC_SYMPTOM_WORDS for word in words)
    return (
        model.symptom_idf.get(symptom, 1.0) - (generic_penalty * 0.2),
        len(words),
        len(symptom),
    )


def pick_informative_symptoms(symptoms: list[str], model: SymptomModel, limit: int = 3) -> list[str]:
    ranked = sorted(symptoms, key=lambda symptom: keyword_score(symptom, model), reverse=True)
    return unique_preserve(ranked)[:limit]


def build_symptom_query(symptoms: list[str], model: SymptomModel) -> tuple[str, str]:
    ordered_start = unique_preserve(symptoms[:2])
    key_symptoms = unique_preserve([*ordered_start, *pick_informative_symptoms(symptoms, model, limit=4)])[:4]
    query_key = " | ".join(key_symptoms[:2]) if key_symptoms[:2] else "unknown symptoms"
    query_text = " ".join(key_symptoms)
    return query_key, query_text


def normalize_search_title(title: str) -> str | None:
    lowered = slugify_disease(title)
    lowered = re.sub(r"^(overview of|evaluation of|approach to|causes of|symptoms of)\s+", "", lowered)
    lowered = re.sub(r"\s+in\s+children and adolescents$", "", lowered)
    if lowered in GENERIC_DISEASE_TITLES or len(lowered) < 4:
        return None
    return lowered


def independent_candidates_from_symptoms(
    symptoms: list[str],
    model: SymptomModel,
    researcher: WebResearcher,
) -> tuple[list[str], list[tuple[str, DiseaseMetrics]], str, list[str]]:
    matches = model.score_row(symptoms)
    ranked = sorted(matches.items(), key=lambda item: (item[1].score, item[1].precision, item[1].recall), reverse=True)
    query_key, query_text = build_symptom_query(symptoms, model)
    search_results = researcher.merck_search(query_text)
    search_candidates = []
    research_titles = []
    symptom_text = " ".join(symptoms).lower()
    prefer_mental_health = any(
        term in symptom_text for term in ("anxiety and nervousness", "depressive or psychotic symptoms", "depression", "insomnia", "fears and phobias")
    )
    for result in search_results:
        url = result["url"].lower()
        title = result["title"]
        if any(token in url for token in ("/conditions/", "/mental-health/conditions/")):
            candidate = normalize_search_title(title)
            if candidate:
                search_candidates.append(candidate)
        normalized_title = normalize_spaces(title)
        if normalized_title.lower().startswith(("symptoms of", "overview -", "treatment -")):
            continue
        if prefer_mental_health and "/mental-health/conditions/" in url:
            research_titles.append(normalized_title)
        elif not prefer_mental_health and "/conditions/" in url:
            research_titles.append(normalized_title)

    if not research_titles:
        for result in search_results:
            normalized_title = normalize_spaces(result["title"])
            if normalized_title.lower().startswith(("symptoms of", "overview -", "treatment -")):
                continue
            research_titles.append(normalized_title)
            if len(research_titles) >= 3:
                break

    candidates: list[str] = []
    for candidate in search_candidates:
        if candidate not in candidates:
            candidates.append(candidate)
    for disease, metrics in ranked[:8]:
        if metrics.precision < 0.28:
            continue
        if len(metrics.matched_symptoms) < 2:
            continue
        if metrics.score < max(0.36, ranked[0][1].score - 0.18):
            continue
        if disease not in candidates:
            candidates.append(disease)
        if len(candidates) >= 6:
            break

    return candidates[:6], ranked, query_key, unique_preserve(research_titles)[:3]


def heuristic_differentials(
    symptoms: list[str],
    label: str,
    ranked: list[tuple[str, DiseaseMetrics]],
) -> list[str]:
    text = " ".join(symptoms).lower()
    candidates: list[str] = []

    def add(*items: str) -> None:
        for item in items:
            normalized = slugify_disease(item)
            if normalized and normalized not in candidates and normalized != label:
                candidates.append(normalized)

    if any(term in text for term in ("anxiety and nervousness", "palpitations", "shortness of breath", "chest tightness", "breathing fast")):
        add("generalised anxiety disorder", "panic attack", "cardiac arrhythmia")
    if any(term in text for term in ("depressive or psychotic symptoms", "depression", "insomnia")):
        add("depression", "bipolar disorder")
    if any(term in text for term in ("palpitations", "irregular heartbeat", "fainting")):
        add("cardiac arrhythmia")
    if any(term in text for term in ("shortness of breath", "cough", "fever", "difficulty breathing")):
        add("pneumonia", "acute bronchitis")
    if any(term in text for term in ("sharp abdominal pain", "upper abdominal pain", "burning abdominal pain", "nausea", "vomiting")):
        add("acute pancreatitis", "cholecystitis", "infectious gastroenteritis")
    if any(term in text for term in ("lower abdominal pain", "blood in urine", "painful urination", "retention of urine")):
        add("urinary tract infection", "cystitis", "kidney stone")
    if any(term in text for term in ("abnormal appearing skin", "itching of skin", "skin irritation", "skin swelling")):
        add("eczema", "psoriasis", "fungal infection of the skin")
    if any(term in text for term in ("pain in eye", "diminished vision", "foreign body sensation in eye", "itchiness of eye")):
        add("conjunctivitis", "corneal abrasion", "uveitis")
    if any(term in text for term in ("sore throat", "throat feels tight", "difficulty in swallowing", "hoarse voice")):
        add("strep throat", "laryngitis", "peritonsillar abscess")
    if any(term in text for term in ("joint pain", "hand or finger pain", "foot or toe pain", "wrist swelling")):
        add("sprain or strain", "gout", "arthritis")

    for disease, metrics in ranked[:8]:
        if disease == label:
            continue
        if metrics.precision < 0.30 or len(metrics.matched_symptoms) < 2:
            continue
        if disease not in candidates:
            candidates.append(disease)
        if len(candidates) >= 5:
            break

    return candidates[:5]


def symptom_presence_phrase(items: list[str]) -> str:
    if not items:
        return "the reported symptom pattern"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def choose_tests(
    symptoms: list[str],
    candidates: list[str],
    disease_payloads: list[dict[str, object]],
) -> list[str]:
    text = " ".join(symptoms).lower()
    candidate_text = " ".join(candidates).lower()
    tests: list[str] = []

    for payload in disease_payloads:
        payload_tests = payload.get("tests", [])
        if isinstance(payload_tests, list):
            tests.extend(test for test in payload_tests if isinstance(test, str))

    def add(*items: str) -> None:
        tests.extend(items)

    if any(term in text for term in ("palpitations", "irregular heartbeat", "chest tightness", "chest pain", "shortness of breath", "breathing fast", "fainting")):
        add("electrocardiogram (ECG)", "pulse oximetry")
        if any(term in text for term in ("irregular heartbeat", "palpitations", "fainting")):
            add("Holter monitoring")
        if any(term in text for term in ("chest pain", "shortness of breath", "cough", "difficulty breathing")):
            add("chest X-ray")
        if any(term in text for term in ("anxiety and nervousness", "palpitations", "increased heart rate", "weight loss")):
            add("thyroid function tests")

    if any(term in text for term in ("cough", "fever", "wheezing", "difficulty breathing", "shortness of breath")):
        add("complete blood count (CBC)")
        if "shortness of breath" in text or "difficulty breathing" in text:
            add("pulse oximetry")

    if any(term in text for term in ("sharp abdominal pain", "burning abdominal pain", "upper abdominal pain", "lower abdominal pain", "nausea", "vomiting")):
        add("complete blood count (CBC)", "comprehensive metabolic panel (CMP)")
        if any(term in text for term in ("upper abdominal pain", "nausea", "vomiting")):
            add("serum lipase")
        if any(term in text for term in ("lower abdominal pain", "retention of urine", "blood in urine")):
            add("urinalysis")
        add("abdominal ultrasound")

    if any(term in text for term in ("blood in urine", "painful urination", "frequent urination", "retention of urine", "symptoms of bladder")):
        add("urinalysis", "urine culture", "renal and bladder ultrasound")

    if any(term in text for term in ("vaginal", "pelvic pain", "painful intercourse", "irregular menstrual", "problem during pregnancy")):
        add("pelvic examination", "pelvic ultrasonography", "pregnancy test")

    if any(term in text for term in ("sore throat", "throat feels tight", "difficulty in swallowing", "hoarse voice", "swollen lymph nodes")):
        add("ENT examination")
        if "sore throat" in text:
            add("rapid strep test")
        if any(term in text for term in ("hoarse voice", "difficulty in swallowing", "throat feels tight")):
            add("laryngoscopy")

    if any(term in text for term in ("pain in eye", "itchiness of eye", "spots or clouds in vision", "foreign body sensation in eye", "diminished vision", "double vision", "cross eyed")):
        add("visual acuity testing", "ophthalmologic examination")
        if any(term in text for term in ("pain in eye", "foreign body sensation in eye")):
            add("slit-lamp examination", "fluorescein staining")

    if any(term in text for term in ("abnormal appearing skin", "itching of skin", "skin irritation", "skin swelling", "irregular appearing scalp", "acne or pimples")):
        add("dermatologic examination")
        if any(term in candidate_text for term in ("fungal", "eczema", "psoriasis")):
            add("skin scraping or fungal culture")
        if any(term in text for term in ("skin swelling", "abnormal appearing skin")):
            add("skin biopsy")

    if any(term in text for term in ("hand or finger pain", "foot or toe pain", "arm pain", "wrist swelling", "injury", "joint pain", "leg pain")):
        add("X-ray")
        if any(term in candidate_text for term in ("arthritis", "gout", "spondylosis", "stenosis")):
            add("joint X-ray")
        if any(term in candidate_text for term in ("complex regional pain syndrome", "concussion", "peripheral nerve disorder")):
            add("MRI of the affected region", "neurologic examination")

    if any(term in text for term in ("depressive or psychotic symptoms", "hostile behavior", "delusions or hallucinations", "anxiety and nervousness", "fears and phobias", "insomnia")):
        add("psychiatric evaluation", "thyroid function tests", "complete blood count (CBC)")
        if any(term in text for term in ("hostile behavior", "delusions or hallucinations", "abusing alcohol", "drug abuse")):
            add("toxicology screening")

    if any(term in text for term in ("increased heart rate", "hypoglycemia", "sweating", "weakness")):
        add("serum glucose")

    return clamp_tests(tests)


def explain_test_relevance(test: str, symptoms: list[str], candidates: list[str]) -> str:
    if test == "electrocardiogram (ECG)":
        return "an electrocardiogram (ECG) is relevant to assess arrhythmia or other cardiac explanations for palpitations, chest symptoms, or dyspnea"
    if test == "Holter monitoring":
        return "Holter monitoring is relevant when intermittent palpitations or irregular heartbeat raise concern for episodic arrhythmia not captured on a single ECG"
    if test == "pulse oximetry":
        return "pulse oximetry is relevant to document oxygenation when shortness of breath or difficulty breathing is reported"
    if test == "chest X-ray":
        return "a chest X-ray is relevant to look for pulmonary or cardiac causes of chest symptoms or breathlessness such as pneumonia or heart failure"
    if test == "thyroid function tests":
        return "thyroid function tests are relevant because hyperthyroid states can mimic anxiety, palpitations, tremulousness, and tachycardia"
    if test == "complete blood count (CBC)":
        return "a complete blood count (CBC) is relevant to screen for infection, inflammation, or anemia that could contribute to the presentation"
    if test == "comprehensive metabolic panel (CMP)":
        return "a comprehensive metabolic panel (CMP) is relevant to assess electrolyte, renal, and hepatic abnormalities that can accompany abdominal or systemic illness"
    if test == "serum lipase":
        return "serum lipase is relevant when upper abdominal pain with nausea or vomiting raises concern for pancreatitis"
    if test == "urinalysis":
        return "urinalysis is relevant to detect hematuria, pyuria, or other urinary findings that help confirm or exclude urinary tract pathology"
    if test == "urine culture":
        return "urine culture is relevant when urinary symptoms suggest an infectious etiology and microbiologic confirmation would guide management"
    if test == "renal and bladder ultrasound":
        return "renal and bladder ultrasound is relevant to evaluate obstruction, retention, stones, or structural urinary abnormalities"
    if test == "pelvic examination":
        return "a pelvic examination is relevant when the symptoms could arise from vulvovaginal, cervical, or other pelvic disease"
    if test == "pelvic ultrasonography":
        return "pelvic ultrasonography is relevant to evaluate uterine, ovarian, adnexal, or pregnancy-related causes of pelvic symptoms"
    if test == "pregnancy test":
        return "a pregnancy test is relevant whenever pelvic, abdominal, or vaginal symptoms could be influenced by pregnancy-related conditions"
    if test == "ENT examination":
        return "an ENT examination is relevant to inspect the pharynx and upper airway when throat pain, tightness, or swallowing difficulty is reported"
    if test == "rapid strep test":
        return "a rapid strep test is relevant to confirm or exclude streptococcal pharyngitis when sore throat and cervical symptoms are present"
    if test == "laryngoscopy":
        return "laryngoscopy is relevant when hoarseness, throat tightness, or dysphagia raises concern for laryngeal or vocal-cord pathology"
    if test == "visual acuity testing":
        return "visual acuity testing is relevant to quantify vision impairment and distinguish mild irritation from more significant ocular disease"
    if test == "ophthalmologic examination":
        return "an ophthalmologic examination is relevant to localize ocular pathology when pain, diplopia, foreign-body sensation, or visual change is reported"
    if test == "slit-lamp examination":
        return "slit-lamp examination is relevant to assess the cornea, anterior chamber, and conjunctiva when eye pain or foreign-body sensation is present"
    if test == "fluorescein staining":
        return "fluorescein staining is relevant to identify corneal abrasion, ulceration, or epithelial injury in a painful or irritated eye"
    if test == "dermatologic examination":
        return "a dermatologic examination is relevant to characterize lesion morphology and distribution before narrowing the skin differential"
    if test == "skin scraping or fungal culture":
        return "skin scraping or fungal culture is relevant when the distribution and texture raise concern for dermatophyte or scalp fungal disease"
    if test == "skin biopsy":
        return "skin biopsy is relevant when the lesion appearance is atypical or persistent and histology is needed to distinguish inflammatory, infectious, or neoplastic causes"
    if test == "X-ray":
        return "an X-ray is relevant to exclude fracture or other bony injury when focal pain or swelling suggests a structural musculoskeletal problem"
    if test == "joint X-ray":
        return "joint X-ray is relevant to assess degenerative or inflammatory joint changes when arthritis-like symptoms are present"
    if test == "MRI of the affected region":
        return "MRI of the affected region is relevant when neuropathic, spinal, or occult soft-tissue pathology remains a serious consideration"
    if test == "neurologic examination":
        return "a neurologic examination is relevant when nerve dysfunction, balance symptoms, or altered sensation could explain the presentation"
    if test == "psychiatric evaluation":
        return "psychiatric evaluation is relevant to distinguish primary panic, mood, psychotic, or substance-related illness from medical mimics"
    if test == "toxicology screening":
        return "toxicology screening is relevant when hallucinations, agitation, or substance-related clues raise concern for intoxication or withdrawal"
    if test == "serum glucose":
        return "serum glucose is relevant when autonomic symptoms, weakness, or possible hypoglycemia could be contributing to the clinical picture"
    if test == "abdominal ultrasound":
        return "abdominal ultrasound is relevant to assess hepatobiliary, renal, or other intra-abdominal causes of abdominal pain"
    return f"{test} is relevant because it helps confirm or exclude the leading differentials suggested by this symptom combination"


def compose_reasoning(
    symptoms: list[str],
    label: str,
    support_status: str,
    candidates: list[str],
    recommended_tests: list[str],
    research_titles: list[str],
    model: SymptomModel,
) -> str:
    label_top_symptoms = model.top_symptoms(label, limit=6)
    supporting = [symptom for symptom in symptoms if symptom in label_top_symptoms][:4]
    weakening = [symptom for symptom in symptoms if symptom not in label_top_symptoms][:3]
    candidate_phrase = symptom_presence_phrase(candidates[:3]) if candidates else "other plausible differentials"

    support_sentence = (
        f"The symptom constellation includes {symptom_presence_phrase(supporting)}, which supports {label} because these features recur prominently in rows labeled as {label}."
        if supporting
        else f"The symptom constellation has only partial overlap with {label}, so the fit is not highly specific."
    )
    weaken_sentence = (
        f"Symptoms such as {symptom_presence_phrase(weakening)} are either nonspecific or broaden the differential, so they weaken a single-diagnosis conclusion."
        if weakening
        else "The remaining features do not introduce a stronger competing pattern than the leading hypothesis."
    )
    research_sentence = (
        f"Symptom-driven web results also surfaced {symptom_presence_phrase(research_titles[:3])}, which reinforces the need to compare this label against overlapping psychiatric, cardiopulmonary, or systemic alternatives."
        if research_titles
        else "The symptom-driven web search did not yield a single definitive condition page for this exact combination, so the interpretation remains conservative."
    )
    test_reasons = " ".join(explain_test_relevance(test, symptoms, candidates) + "." for test in recommended_tests[:3])

    if support_status == "supported":
        think = (
            f"<think>{support_sentence} {weaken_sentence} "
            f"{research_sentence} "
            f"No alternative diagnosis explains the overall pattern as well as {label}, given the balance of symptom overlap and the lack of a comparably strong competing pattern. "
            f"{test_reasons}</think>"
        )
        conclusion = (
            f"Based on the reported symptoms, the clinical indication most strongly points to: {label}, "
            f"although confirmatory evaluation with {symptom_presence_phrase(recommended_tests[:3])} is advisable."
        )
        return f"{think}\n{conclusion}\n{DISCLAIMER}"

    think = (
        f"<think>{support_sentence} {weaken_sentence} "
        f"{research_sentence} "
        f"The symptom-based differential remains broad enough that {label} cannot be confirmed with confidence; plausible competing diagnoses include {candidate_phrase}. "
        f"{test_reasons}</think>"
    )
    if candidates:
        conclusion = (
            f"Based on the reported symptoms, it is not possible to confirm a single diagnosis with confidence; "
            f"the case may be {candidate_phrase}, and the most relevant next steps include {symptom_presence_phrase(recommended_tests[:3])}."
        )
    else:
        conclusion = (
            f"Based on the reported symptoms, it is not possible to confirm a single diagnosis with confidence, and the symptom set remains clinically inconclusive; "
            f"the most relevant next steps include {symptom_presence_phrase(recommended_tests[:3])}."
        )
    return f"{think}\n{conclusion}\n{DISCLAIMER}"


def decide_row(
    model: SymptomModel,
    researcher: WebResearcher,
    row: dict[str, str],
) -> tuple[str, str, str, str]:
    label = slugify_disease(row["output"])
    symptoms = model.parse_symptoms(row["input"])

    independent_candidates, ranked, _query_key, research_titles = independent_candidates_from_symptoms(symptoms, model, researcher)
    matches = model.score_row(symptoms)
    label_metrics = matches.get(label, DiseaseMetrics(0.0, 0.0, 0.0, tuple(), 0.0))
    top_disease, top_metrics = ranked[0] if ranked else (label, label_metrics)
    second_metrics = ranked[1][1] if len(ranked) > 1 else DiseaseMetrics(0.0, 0.0, 0.0, tuple(), 0.0)
    supported = (
        top_disease == label
        and label_metrics.score >= 0.56
        and label_metrics.precision >= 0.48
        and label_metrics.recall >= 0.65
        and (top_metrics.score - second_metrics.score) >= 0.10
        and len(symptoms) >= 2
    )

    if supported:
        support_status = "supported"
        candidate_diseases = [label]
        disease_payloads = [researcher.disease_page(label)]
    else:
        support_status = "inconclusive"
        heuristic_candidates = heuristic_differentials(symptoms, label, ranked)
        ranked_candidates = [disease for disease, metrics in ranked if metrics.precision >= 0.30 and len(metrics.matched_symptoms) >= 2][:5]
        merged = unique_preserve([*heuristic_candidates, *ranked_candidates, *independent_candidates])
        candidate_diseases = [slugify_disease(candidate) for candidate in merged if slugify_disease(candidate) and slugify_disease(candidate) != label][:3]
        disease_payloads = [researcher.disease_page(disease) for disease in unique_preserve([label, *candidate_diseases])[:3]]

    recommended_tests = choose_tests(symptoms, candidate_diseases or [label], disease_payloads)
    reasoning = compose_reasoning(
        symptoms=symptoms,
        label=label,
        support_status=support_status,
        candidates=candidate_diseases,
        recommended_tests=recommended_tests,
        research_titles=research_titles,
        model=model,
    )
    return (
        support_status,
        json_dumps(candidate_diseases if support_status == "inconclusive" else [label]),
        json_dumps(recommended_tests),
        reasoning,
    )


def count_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def count_processed_rows(output_csv: Path) -> int:
    if not output_csv.exists():
        return 0
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def summarize_output_rows(output_csv: Path) -> dict[str, int]:
    summary = {
        "completed_rows": 0,
        "supported_rows": 0,
        "inconclusive_rows": 0,
    }
    if not output_csv.exists():
        return summary

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            summary["completed_rows"] += 1
            support_status = row.get("support_status")
            if support_status == "supported":
                summary["supported_rows"] += 1
            elif support_status == "inconclusive":
                summary["inconclusive_rows"] += 1
    return summary


def load_progress() -> dict:
    if not PROGRESS_JSON.exists():
        return {
            "input_file": str(INPUT_CSV),
            "output_file": str(OUTPUT_CSV),
            "batch_size": BATCH_SIZE,
            "completed_rows": 0,
            "supported_rows": 0,
            "inconclusive_rows": 0,
            "error_rows": [],
            "started_from_zero": True,
        }
    return json.loads(PROGRESS_JSON.read_text(encoding="utf-8"))


def write_progress(payload: dict) -> None:
    PROGRESS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_outputs(input_csv: Path, output_csv: Path) -> dict[str, object]:
    original_rows = []
    enriched_rows = []

    with input_csv.open("r", encoding="utf-8", newline="") as src_handle:
        original_rows = list(csv.DictReader(src_handle))
    with output_csv.open("r", encoding="utf-8", newline="") as out_handle:
        enriched_rows = list(csv.DictReader(out_handle))

    row_order_ok = len(original_rows) == len(enriched_rows)
    missing_support = 0
    invalid_recommended = 0
    invalid_reasoning = 0

    for original, enriched in zip(original_rows, enriched_rows):
        if any(original[field] != enriched[field] for field in ("input", "output", "source")):
            row_order_ok = False
        if enriched.get("support_status") not in {"supported", "inconclusive"}:
            missing_support += 1
        try:
            parsed = json.loads(enriched.get("recommended_exams_tests", ""))
            if not isinstance(parsed, list) or not parsed:
                invalid_recommended += 1
        except json.JSONDecodeError:
            invalid_recommended += 1
        reasoning = enriched.get("reasoning", "")
        if not reasoning.startswith("<think>") or DISCLAIMER not in reasoning:
            invalid_reasoning += 1

    return {
        "original_rows": len(original_rows),
        "enriched_rows": len(enriched_rows),
        "row_order_ok": row_order_ok,
        "missing_support_status": missing_support,
        "invalid_recommended_exams_rows": invalid_recommended,
        "invalid_reasoning_rows": invalid_reasoning,
    }


def reset_outputs() -> None:
    for path in (OUTPUT_CSV, PROGRESS_JSON, RESEARCH_CACHE_JSON):
        if path.exists():
            path.unlink()


def process_batches(limit_batches: int | None = None, start_over: bool = False) -> dict[str, object]:
    if start_over:
        reset_outputs()

    model = SymptomModel()
    model.build(INPUT_CSV)
    researcher = WebResearcher(RESEARCH_CACHE_JSON)
    progress = load_progress()
    output_summary = summarize_output_rows(OUTPUT_CSV)
    completed_rows = output_summary["completed_rows"]
    progress["completed_rows"] = completed_rows
    progress["supported_rows"] = output_summary["supported_rows"]
    progress["inconclusive_rows"] = output_summary["inconclusive_rows"]
    progress["total_rows"] = count_rows(INPUT_CSV)

    write_header = not OUTPUT_CSV.exists()
    batch_counter = 0

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as input_handle, OUTPUT_CSV.open(
        "a", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(output_handle, fieldnames=OUTPUT_FIELDS)

        if write_header:
            writer.writeheader()
            output_handle.flush()

        for _ in range(completed_rows):
            next(reader, None)

        batch_rows: list[dict[str, str]] = []
        current_index = completed_rows

        def flush_batch_rows(force_cache_save: bool = False) -> None:
            if not batch_rows:
                return
            writer.writerows(batch_rows)
            output_handle.flush()
            batch_rows.clear()
            progress["completed_rows"] = current_index
            write_progress(progress)
            if force_cache_save:
                researcher.save_cache()
            else:
                researcher.maybe_save_cache()

        try:
            for row in reader:
                current_index += 1
                try:
                    support_status, candidate_diseases, recommended_exams_tests, reasoning = decide_row(model, researcher, row)
                    enriched_row = {
                        "input": row["input"],
                        "output": row["output"],
                        "source": row["source"],
                        "support_status": support_status,
                        "candidate_diseases": candidate_diseases,
                        "recommended_exams_tests": recommended_exams_tests,
                        "reasoning": reasoning,
                    }
                    if support_status == "supported":
                        progress["supported_rows"] = progress.get("supported_rows", 0) + 1
                    else:
                        progress["inconclusive_rows"] = progress.get("inconclusive_rows", 0) + 1
                except Exception as exc:  # pragma: no cover - safety fallback
                    progress.setdefault("error_rows", []).append(current_index)
                    progress["inconclusive_rows"] = progress.get("inconclusive_rows", 0) + 1
                    enriched_row = {
                        "input": row["input"],
                        "output": row["output"],
                        "source": row["source"],
                        "support_status": "inconclusive",
                        "candidate_diseases": json_dumps([]),
                        "recommended_exams_tests": json_dumps([
                            "complete blood count (CBC)",
                            "basic metabolic panel (BMP)",
                        ]),
                        "reasoning": (
                            f"<think>An internal processing issue prevented a reliable row-specific interpretation for this symptom set. "
                            f"A complete blood count (CBC) is relevant to screen for infection, inflammation, or anemia, and a basic metabolic panel (BMP) is relevant to detect metabolic derangements while the case is manually reviewed. "
                            f"Processing error: {type(exc).__name__}.</think>\n"
                            "Based on the reported symptoms, it is not possible to confirm a single diagnosis with confidence, and the symptom set remains clinically inconclusive; the most relevant next steps include complete blood count (CBC) and basic metabolic panel (BMP).\n"
                            f"{DISCLAIMER}"
                        ),
                    }

                batch_rows.append(enriched_row)

                if len(batch_rows) == BATCH_SIZE:
                    flush_batch_rows()
                    batch_counter += 1
                    if limit_batches is not None and batch_counter >= limit_batches:
                        break

            if batch_rows and (limit_batches is None or batch_counter < limit_batches):
                flush_batch_rows(force_cache_save=True)
        except KeyboardInterrupt:
            flush_batch_rows(force_cache_save=False)
            raise

    validation = validate_outputs(INPUT_CSV, OUTPUT_CSV) if progress["completed_rows"] == progress["total_rows"] else None
    progress["validation"] = validation
    write_progress(progress)
    researcher.save_cache()
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable clinical CSV enrichment processor with live web research cache")
    parser.add_argument("--limit-batches", type=int, default=None, help="Process only a fixed number of batches.")
    parser.add_argument("--start-over", action="store_true", help="Delete previous outputs and start from zero.")
    args = parser.parse_args()
    result = process_batches(limit_batches=args.limit_batches, start_over=args.start_over)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
