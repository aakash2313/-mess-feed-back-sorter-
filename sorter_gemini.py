"""Mess Feedback Sorter: LLM classification with strict validation and safety override."""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

CATEGORIES = {'taste', 'hygiene', 'quantity', 'timing', 'other'}
SENTIMENTS = {'positive', 'negative', 'neutral'}
REQUIRED = {'category', 'sentiment', 'urgent', 'summary'}
# Extend this list after review with Tamil-speaking mess staff.
RISK_PATTERNS = [
    r'\binsects?\b', r'\bbugs?\b', r'\bcockroaches?\b', r'\bflies?\b',
    r'\bworms?\b', r'\bmaggots?\b', r'\bhair\b', r'\bforeign\s+objects?\b',
    r'\bglass\b', r'\bplastic\s+(?:piece|pieces|in food)\b', r'\bmetal\s+(?:piece|pieces|in food)\b',
    r'\bexpired\b', r'\bspoiled\b', r'\brotten\b', r'\bstale\b', r'\bmold(?:y)?\b',
    r'\bmould(?:y)?\b', r'\bfood\s+poisoning\b', r'\bfell\s+sick\b',
    r'\bgot\s+sick\b', r'\bvomit(?:ed|ing)?\b', r'\bdiarrh(?:ea|oea)\b',
    r'\bpoochi\b', r'\bpuchi\b', r'\bmudi\b', r'\bketta\s+(?:saapadu|soru|food)\b',
    r'பூச்சி', r'முடி', r'கெட்ட\s*சாப்பாடு', r'வாந்தி', r'வயிற்றுப்போக்கு',
]
RISK_RE = re.compile('|'.join(f'(?:{p})' for p in RISK_PATTERNS), re.IGNORECASE)
SYSTEM_PROMPT = '''Classify hostel mess feedback, including English, Tamil and Tanglish.
Return ONLY a JSON object with EXACTLY these keys: category, sentiment, urgent, summary.
category must be one of taste, hygiene, quantity, timing, other.
sentiment must be one of positive, negative, neutral.
urgent must be a JSON boolean, true only for a health risk: foreign object, stale/spoiled/expired food, or someone falling sick.
Taste, quantity and timing complaints are never urgent merely because of angry wording.
summary must be a nonempty string of at most 12 whitespace-separated words.
No markdown or extra fields. Do not invent facts.'''


def safety_match(comment):
    match = RISK_RE.search(comment)
    return match.group(0) if match else ''


def validate(raw):
    """Return (validated dict, error). Extra/missing fields and wrong types fail."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return None, 'invalid JSON'
    if not isinstance(data, dict) or set(data) != REQUIRED:
        return None, 'wrong field names or number of fields'
    if type(data['category']) is not str or data['category'] not in CATEGORIES:
        return None, 'invalid category'
    if type(data['sentiment']) is not str or data['sentiment'] not in SENTIMENTS:
        return None, 'invalid sentiment'
    if type(data['urgent']) is not bool:
        return None, 'urgent must be boolean'
    if type(data['summary']) is not str or not data['summary'].strip() or len(data['summary'].split()) > 12:
        return None, 'summary must be 1 to 12 words'
    return data, None


def api_classifier(comment, model):
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY is not set; use --mode demo for offline testing')
    payload = {'model': model, 'temperature': 0,
               'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                            {'role': 'user', 'content': comment}]}
    request = urllib.request.Request('https://api.openai.com/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        return result['choices'][0]['message']['content']
    except (urllib.error.URLError, KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f'API request failed: {exc}') from exc



def gemini_classifier(comment, model):
    """Classify with Google Gemini using the GEMINI_API_KEY environment variable."""
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise RuntimeError('GEMINI_API_KEY is not set')
    payload = {
        'system_instruction': {'parts': [{'text': SYSTEM_PROMPT}]},
        'contents': [{'role': 'user', 'parts': [{'text': comment}]}],
        'generationConfig': {'responseMimeType': 'application/json'},
    }
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'x-goog-api-key': key, 'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        return ''.join(part.get('text', '') for part in result['candidates'][0]['content']['parts'])
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')[:350]
        raise RuntimeError(f'Gemini HTTP {exc.code}: {detail}') from exc
    except (urllib.error.URLError, KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f'Gemini request failed: {exc}') from exc

def demo_classifier(comment, model=None):
    """Deterministic offline stand-in; NOT an LLM. Useful for running sample project."""
    c = comment.casefold()
    risk = safety_match(comment)
    if risk or any(w in c for w in ['clean', 'dirty', 'plate', 'கழுவ']):
        category = 'hygiene'
    elif any(w in c for w in ['late', 'queue', 'time', 'delay', 'நேரம்']):
        category = 'timing'
    elif any(w in c for w in ['portion', 'quantity', 'less rice', 'serving', 'போதவில்லை']):
        category = 'quantity'
    elif any(w in c for w in ['salt', 'spicy', 'taste', 'tasty', 'bland', 'cold', 'undercooked', 'சுவை', 'காரம்']):
        category = 'taste'
    else:
        category = 'other'
    if any(w in c for w in ['good', 'great', 'nice', 'tasty', 'thank', 'excellent', 'நல்ல']):
        sentiment = 'positive'
    elif risk or any(w in c for w in ['bad', 'too ', 'less', 'late', 'dirty', 'cold', 'undercooked', 'not ', 'again', 'delay', 'போதவில்லை']):
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return json.dumps({'category': category, 'sentiment': sentiment,
                       'urgent': bool(risk), 'summary': ' '.join(comment.split()[:12])}, ensure_ascii=False)


def process(input_file, output_dir, classifier, model):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_ids = set()
    with input_file.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or set(reader.fieldnames) != {'id', 'date', 'meal', 'comment'} or len(reader.fieldnames) != 4:
            raise ValueError('Input CSV must have exactly id,date,meal,comment columns')
        for line, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f'Extra CSV columns at line {line}')
            feedback_id = row['id'].strip()
            if not feedback_id or feedback_id in seen_ids:
                raise ValueError(f'Missing or duplicate id at line {line}: {feedback_id!r}')
            seen_ids.add(feedback_id)
            comment = row['comment'] or ''
            result = {'id': feedback_id, 'date': row['date'], 'meal': row['meal'],
                      'comment': comment, 'category': 'needs_manual_review',
                      'sentiment': 'needs_manual_review', 'urgent': False,
                      'summary': '', 'status': 'needs_manual_review', 'safety_keyword': '', 'review_reason': ''}
            risk = safety_match(comment)
            result['safety_keyword'] = risk
            if not comment.strip():
                result['review_reason'] = 'empty comment'
            else:
                for attempt in range(2):
                    try:
                        raw = classifier(comment, model)
                        valid, error = validate(raw)
                    except Exception as exc:
                        valid, error = None, f'classifier error: {exc}'
                    if valid is not None:
                        result.update(valid)
                        result['status'] = 'classified'
                        break
                    result['review_reason'] = f'failed after {attempt + 1} attempt(s): {error}'
                if result['status'] == 'classified':
                    result['review_reason'] = ''
            # Always apply after validation/retries, including manual-review cases.
            # A safety keyword can only escalate; never clear an LLM urgent flag.
            if risk:
                result['urgent'] = True
            rows.append(result)
    columns = ['id', 'date', 'meal', 'comment', 'category', 'sentiment', 'urgent',
               'summary', 'status', 'safety_keyword', 'review_reason']
    def write_csv(path, items):
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for item in items:
                writer.writerow({**item, 'urgent': str(item['urgent']).lower()})
    write_csv(output_dir / 'classified_feedback.csv', rows)
    urgent = [r for r in rows if r['urgent']]
    write_csv(output_dir / 'urgent_for_warden.csv', urgent)
    counts = Counter(r['category'] for r in rows)
    sentiments = Counter(r['sentiment'] for r in rows)
    review = [r for r in rows if r['status'] == 'needs_manual_review']
    with (output_dir / 'committee_summary.txt').open('w', encoding='utf-8') as f:
        f.write('MESS FEEDBACK COMMITTEE SUMMARY\n' + '=' * 32 + '\n')
        f.write(f'Total comments: {len(rows)}\nUrgent flagged: {len(urgent)}\nNeeds manual review: {len(review)}\n\n')
        f.write('Categories:\n')
        for name in sorted(CATEGORIES) + ['needs_manual_review']:
            f.write(f'  {name}: {counts[name]}\n')
        f.write('\nSentiments:\n')
        for name in sorted(SENTIMENTS) + ['needs_manual_review']:
            f.write(f'  {name}: {sentiments[name]}\n')
        f.write('\nUrgent IDs: ' + (', '.join(r['id'] for r in urgent) or 'none') + '\n')
        f.write('Manual-review IDs: ' + (', '.join(r['id'] for r in review) or 'none') + '\n')
        f.write('\nHuman review required: automated flags are triage only, not confirmed incidents.\n')
        f.write('Keyword matching cannot guarantee detection of all health risks.\n')
    return len(rows), len(urgent), len(review)


def main():
    parser = argparse.ArgumentParser(description='Sort hostel mess feedback')
    parser.add_argument('--input', type=Path, default=Path('feedback.csv'))
    parser.add_argument('--output', type=Path, default=Path('output'))
    parser.add_argument('--mode', choices=['api', 'gemini', 'demo'], default='api',
                        help='api = OpenAI; gemini = Google Gemini; demo = offline stand-in')
    parser.add_argument('--model', default=None,
                        help='Optional model override; defaults depend on mode')
    args = parser.parse_args()
    if args.model is None:
        args.model = 'gemini-3.5-flash' if args.mode == 'gemini' else 'gpt-4o-mini'
    if args.mode == 'gemini' and not os.environ.get('GEMINI_API_KEY'):
        parser.error('Set GEMINI_API_KEY for --mode gemini')
    if args.mode == 'api' and not os.environ.get('OPENAI_API_KEY'):
        parser.error('Set OPENAI_API_KEY for --mode api or use --mode demo')
    try:
        total, urgent, review = process(args.input, args.output,
                                         {'api': api_classifier, 'gemini': gemini_classifier,
                                          'demo': demo_classifier}[args.mode], args.model)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'Mode: {args.mode} | Comments: {total} | Urgent: {urgent} | Manual review: {review}')
    print(f'Wrote 3 files to {args.output.resolve()}')


if __name__ == '__main__':
    main()
