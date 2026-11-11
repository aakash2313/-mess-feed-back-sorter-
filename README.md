# Mess Feedback Sorter

CIT AI-Native Engineering Programme · Semester 1 · Assessment 5

A console-based Python application that reads `feedback.csv`, classifies each comment using an LLM, validates every response, escalates known food-safety keywords, and produces exactly three output files. **A human decides what to do with the feedback.**

## Requirements

Python 3.9+; no third-party Python packages required. Internet access and an OpenAI API key are required for real LLM mode. API usage may incur charges.

## Run

From this folder:

```bash
# Offline demonstration (NOT an LLM; produces example output files)
python sorter.py --mode demo

# Real LLM mode (macOS/Linux)
export OPENAI_API_KEY="your-api-key"
python sorter.py --mode api

# Real LLM mode (Windows PowerShell)
$env:OPENAI_API_KEY="your-api-key"
python sorter.py --mode api

# Optional custom paths/model
python sorter.py --mode api --input feedback.csv --output output --model gpt-4o-mini

# Run tests
python -m unittest -v test_sorter.py
```

Do not commit or share your API key. The default mode is `api`, so the actual project uses an LLM unless you explicitly request the offline demo.

## Input

`feedback.csv` must have exactly `id,date,meal,comment`. The sample has 24 rows in English, Tamil, and Tanglish, including one empty comment. IDs must be unique and nonempty.

## Output (exactly three files)

- `output/classified_feedback.csv`: all feedback with LLM fields, review status, safety keyword, and reason.
- `output/urgent_for_warden.csv`: all urgent feedback, including safety-overridden manual-review records.
- `output/committee_summary.txt`: counts by category and sentiment, urgent IDs, and manual-review IDs.

## Rules and design

1. Empty or whitespace-only comments are never sent to the LLM; they require manual review.
2. LLM output must be a JSON object with exactly `category`, `sentiment`, `urgent`, `summary`; category and sentiment must be allowed strings, urgent must be a real boolean, summary 1–12 words.
3. On any invalid response or API error, retry **once** (two attempts total). If both fail, mark category/sentiment/status `needs_manual_review`.
4. Apply the keyword safety net **after** classification/retries. Matching words force urgent to `true`, even for manual-review rows. The safety net never changes `true` to `false`.
5. Taste, timing, and quantity complaints are not urgent solely due to emotion. The LLM prompt instructs this; the safety net only escalates explicit risk words.
6. Do not treat outputs as confirmed incidents. Keyword matching is not exhaustive, especially across languages, spelling variants, and negation (e.g. "no insect"). Humans should review flagged records and any suspicious unflagged feedback.

## Architecture

```text
feedback.csv
    |
    v
CSV reader / ID validation
    |
    +-- empty comment ----------------> manual review
    |
    +-- nonempty --> LLM --> strict validation --> retry once if invalid
                                              |
                                              +--> manual review if still invalid
    |
    v
Keyword safety escalation (never de-escalation)
    |
    +--> classified_feedback.csv
    +--> urgent_for_warden.csv
    +--> committee_summary.txt
```

## Demonstration vs. real AI

The included sample output is generated in `--mode demo` by simple deterministic rules, not by an LLM. Use `--mode api` with a real API key to demonstrate the actual AI classification requirement. Real-model classifications may differ. The output CSVs contain the final, safety-checked results.

## Suggested showcase demonstration

Show an insect complaint and a Tamil `பூச்சி` complaint becoming urgent; an empty comment going to manual review without a model call; an invalid JSON response being retried once (covered by the unit tests); and an LLM returning `urgent:false` for an insect complaint that Python overrides to `true` (also covered by the tests).
