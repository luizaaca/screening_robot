# Prompt for batch processing

You are a clinical dataset validation assistant.
Assess whether the reported symptoms strongly support the provided disease label.
Decide whether the symptoms allow identifying this disease with confidence, or whether the case is inconclusive. Analyze each case independently, reason critically about the symptom constellation, and suggest targeted exams/tests that would help a physician confirm or exclude the main hypotheses.


# Goal

- Read the file `combined_diseases_symptoms_2.csv`.
- Process iteratively in batches of 10 rows at a time.
- First analyze only the `input` column for each row and reason independently about the possible diagnoses.
- Use the `output` column only after the independent symptom-based reasoning, to decide whether the labeled disease is actually supported or whether the case remains inconclusive.
- Ignore the `source` column for the clinical analysis.
- For each row, perform real web research using `fetch_webpage`: build a symptom-based search query such as `symptoms: <keyword-1, keyword-2, keyword-3>` and use it to fetch a search URL or authoritative medical pages; then recursively fetch the most relevant linked pages until you have enough evidence to justify the likely diseases and recommended exams/tests.
- Prefer authoritative clinical references such as MedlinePlus, NIH, NHS, Mayo Clinic, Cleveland Clinic, MSD/Merck Manual, professional guidelines, or similarly reliable medical sources.
- Produce a new enriched CSV.
- Save the result incrementally after each batch so the process is resumable.
- Keep each result exactly at the same position as the original row to allow a subsequent join between the original and the processed file.


# Critical non-generic requirement

- The examples below define structure only. Do not copy or paraphrase them mechanically.
- Every `reasoning` block must be freshly written from the actual symptom pattern of that row and the fetched web evidence.
- Explicitly discuss which symptoms support a hypothesis, which symptoms weaken it, and which symptoms require ruling out alternative diagnoses.
- Explicitly explain why each recommended exam/test is relevant for confirming or excluding the leading diagnoses.
- Do not write vague filler such as `tests may help` without naming concrete exams/tests whenever reasonably possible.
- Do not invent web findings. If the evidence remains weak or conflicting after research, mark the row as `inconclusive`.


# Examples

Input rows (these are the rows as they appear in the original CSV, `input,output,source`, you must analyze each row independently and produce the corresponding output columns `support_status,candidate_diseases,recommended_exams_tests,reasoning`):

```csv
input,output,source
"anxiety and nervousness, shortness of breath, depressive or psychotic symptoms, chest tightness, palpitations, irregular heartbeat, breathing fast",panic disorder,ds1_diseases_symptoms
"shortness of breath, depressive or psychotic symptoms, dizziness, insomnia, palpitations",panic disorder,ds1_diseases_symptoms
```

Corresponding expected output rows (these are the columns to be inserted into the enriched CSV: `support_status,candidate_diseases,recommended_exams_tests,reasoning`), in the same order as the input rows above:

```csv
support_status,candidate_diseases,recommended_exams_tests,reasoning
supported,"[\"panic disorder\"]","[\"electrocardiogram (ECG)\", \"thyroid function tests\"]","<think>The symptom cluster includes anxiety and nervousness, chest tightness, palpitations, irregular heartbeat, shortness of breath, and rapid breathing, which together strongly fit panic-spectrum presentations and therefore support panic disorder. However, palpitations and irregular heartbeat can also occur in cardiac or endocrine conditions, so an electrocardiogram (ECG) is relevant to evaluate arrhythmia and thyroid function tests are relevant to exclude hyperthyroid states that may mimic panic symptoms.</think>\nBased on the reported symptoms, the clinical indication most strongly points to: panic disorder, although confirmatory evaluation with electrocardiogram (ECG) and thyroid function tests is advisable.\nDisclaimer: This is an AI auxiliary tool designed for healthcare professionals. It is not 100% precise and does not replace a professional medical diagnosis."
inconclusive,"[\"depression\", \"bipolar disorder\"]","[\"electrocardiogram (ECG)\", \"complete blood count (CBC)\", \"psychiatric evaluation\"]","<think>The combination of shortness of breath, dizziness, and palpitations can overlap with panic symptoms, but the concurrent depressive or psychotic symptoms and insomnia broaden the differential toward mood disorders such as depression or bipolar disorder. An electrocardiogram (ECG) is relevant to assess cardiopulmonary mimics of palpitations and dyspnea, a complete blood count (CBC) is useful to screen for systemic contributors such as anemia, and psychiatric evaluation is relevant to distinguish primary mood illness from anxiety-driven symptom clusters.</think>\nBased on the reported symptoms, it is not possible to confirm a single diagnosis with confidence; the case may be depression or bipolar disorder, and the most relevant next steps include electrocardiogram (ECG), complete blood count (CBC), and psychiatric evaluation.\nDisclaimer: This is an AI auxiliary tool designed for healthcare professionals. It is not 100% precise and does not replace a professional medical diagnosis."
```


# Required web research workflow

For each row, do all of the following before finalizing the output:

1. Extract the 2 to 6 most clinically informative symptom keywords from `input`.
2. Form a symptom-driven query such as `symptoms: anxiety and nervousness, chest tightness, palpitations, shortness of breath`.
3. Use `fetch_webpage` to open a search URL or authoritative medical reference URL based on that query.
4. Recursively fetch the most relevant linked pages until you have enough evidence to justify:
   - the most plausible diagnosis or differential diagnoses;
   - the most relevant exams/tests to confirm or exclude them.
5. Prefer high-quality medical sources and synthesize the findings in your own words. Do not copy source text.
6. If the sources disagree or the symptom set remains too broad, stay conservative and mark the row as `inconclusive`.
7. The final `reasoning` and `recommended_exams_tests` must clearly reflect the fetched evidence, not just the symptom list alone.

# Processing rules

0. Use subagents to preserve context-length of main session, use session memory to keep track of progress, and use `fetch_webpage` for real web research as described above.
1. The input file has columns: `input`, `output`, `source`.
2. For clinical analysis:
   - `input` = reported symptoms.
   - `output` = labeled disease/diagnosis, to be compared only after the independent symptom-based reasoning is complete.
   - `source` is irrelevant for the clinical analysis.
3. For each row, evaluate the symptoms, use symptom-by-symptom clinical reasoning, perform web research, and provide a brief but specific reasoning for your assessment. You must be critical and conservative in your evaluation, consider alternative diagnoses when applicable, and recommend specific examinations or tests.
4. The result must add to the new CSV exactly these columns:
   - `support_status`
   - `candidate_diseases`
   - `recommended_exams_tests`
   - `reasoning`
5. Also preserve the original columns `input`, `output`, and `source` in the output CSV.
6. Do not remove, reorder, filter, or deduplicate rows.
7. Do not alter the original text of `input`, `output`, or `source`. Do not edit any other file in the workspace.
8. After completing each batch of 10 rows, immediately write the partial CSV to disk before continuing to the next batch.
9. Processing must be iterative and resumable. If a partial/output file already exists with processed rows, continue from the next pending row.
10. If an error occurs on a row, do not stop the whole batch. Record the error on that row and continue.
11. Do not use canned differential diagnoses or canned exams/tests. The candidate diseases and recommended exams/tests must be justified by the row-specific symptom profile and the fetched medical information.


# Clinical criteria to apply per row

- Set `support_status = "supported"` only when the symptoms support the provided diagnosis with confidence, the fetched evidence is compatible with that diagnosis, and there is no other plausible alternative that explains the symptom set comparably well.
- Set `support_status = "inconclusive"` when the presentation is not conclusive, ambiguous, or fit more than one disease.
- When `support_status = "supported"`, `candidate_diseases` should contain only the disease from the `output` column.
- When `support_status = "inconclusive"`, `candidate_diseases` should contain up to 3 short, plausible differential diagnoses grounded in the symptoms and web research findings.
- `candidate_diseases` must be written to the CSV as a JSON-serialized list.
- `recommended_exams_tests` must contain 1 to 5 concrete exams/tests that help confirm or exclude the leading hypotheses for that row.
- Prefer specific tests such as `electrocardiogram (ECG)`, `chest X-ray`, `urinalysis`, `complete blood count (CBC)`, `thyroid function tests`, `laryngoscopy`, `pelvic examination`, `ophthalmologic examination`, etc., instead of vague phrases such as `blood tests` whenever more specific naming is possible.
- `recommended_exams_tests` must be written to the CSV as a JSON-serialized list.


# Format for the `reasoning` text

- The `reasoning` field must follow this exact structure:
  1. A `<think>...</think>` block with actual clinical reasoning about the symptoms, the main competing hypotheses, and why the recommended exams/tests are relevant.
  2. A final conclusion in natural language in a new line, written specifically for that row and mentioning the most relevant next exams/tests.
  3. A new line with the disclaimer below.
- The `reasoning` text must not be a generic restatement of the examples; it must be tailored to the row being analyzed.


# Examples on how to compose `reasoning`

- If the case is supported, the conclusion should generally follow this pattern while remaining row-specific:
  `Based on the reported symptoms, the clinical indication most strongly points to: <disease>, although confirmatory evaluation with <test(s)> is advisable.`
- If the case is inconclusive and there are alternatives:
  `Based on the reported symptoms, it is not possible to confirm a single diagnosis with confidence; the case may be <alternative(s)>, and the most relevant next steps include <test(s)>.`
- If the case is inconclusive and there are no alternatives:
  `Based on the reported symptoms, it is not possible to confirm a single diagnosis with confidence, and the symptom set remains clinically inconclusive; the most relevant next steps include <test(s)>.`
- Always end with this exact disclaimer:
  `Disclaimer: This is an AI auxiliary tool designed for healthcare professionals. It is not 100% precise and does not replace a professional medical diagnosis.`


# Expected normalization

- Normalize the disease name to lowercase when building derived fields.
- Remove duplicates from `candidate_diseases`.
- Remove duplicates from `recommended_exams_tests`.
- If `support_status = "supported"`, ensure `candidate_diseases` is a list containing only the `output` disease in lowercase.
- Ensure `recommended_exams_tests` is always a JSON list and is never empty.
- Keep exam/test names in conventional clinical naming/capitalization.


# Expected implementation
- Create a new output file, for example: `combined_diseases_symptoms_2_enriched_with_exams.csv`.
- Also create, if needed, an auxiliary progress/log file to support resuming batch-by-batch processing.
- Persist to disk at the end of each batch of 10 rows.
- Before processing the next batch, confirm how many rows have already been completed.
- If an older partial file exists but does not contain the new column `recommended_exams_tests`, create a new versioned output file instead of corrupting the old schema.
- At the end, validate that:
  - the number of rows in the enriched file equals the original file;
  - the row order remains identical to the original file;
  - all rows have `support_status` filled;
  - all rows have `recommended_exams_tests` filled with valid JSON lists;
  - all rows have `reasoning` starting with `<think>` and containing the required disclaimer.


# Expected output of your work

- Run the batches iteratively.
- Update the output CSV after each batch.
- When finished, report:
  - the generated file name;
  - total number of rows processed;
  - number of `supported` rows;
  - number of `inconclusive` rows;
  - whether there were errors and which rows they affected.


If you need to create a helper Python script to do this safely and resumably, you may. Prefer a robust, incremental, and idempotent implementation.
