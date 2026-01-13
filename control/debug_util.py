#!/usr/bin/env python
"""Debug utility for analyzing exploration data."""

import json
import re
import sys
from pathlib import Path

FANO_ROOT = Path(__file__).parent.parent
EXPLORATIONS_DIR = FANO_ROOT / "explorer" / "data" / "explorations"


def analyze_math_patterns(thread_id: str, exchange_idx: int = None):
    """Analyze math delimiter patterns in a thread."""
    path = EXPLORATIONS_DIR / f"{thread_id}.json"
    if not path.exists():
        print(f"Thread not found: {thread_id}")
        return

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    exchanges = data.get('exchanges', [])
    if exchange_idx is not None:
        exchanges = [exchanges[exchange_idx]] if exchange_idx < len(exchanges) else []

    for i, ex in enumerate(exchanges):
        resp = ex.get('response', '')
        print(f"\n=== Exchange {i}: {ex.get('role')} ({ex.get('model')}) ===")

        # Count different delimiter patterns
        patterns = {
            r'\\(': 'backslash-paren \\(',
            r'\\)': 'backslash-paren \\)',
            r'\\[': 'backslash-bracket \\[',
            r'\\]': 'backslash-bracket \\]',
            r'\$\$': 'double-dollar $$',
            r'(?<!\$)\$(?!\$)': 'single-dollar $',
        }

        print("Delimiter counts:")
        for pat, name in patterns.items():
            try:
                count = len(re.findall(pat, resp))
                if count > 0:
                    print(f"  {name}: {count}")
            except:
                pass

        # Find "( \command )" patterns (malformed math)
        malformed = re.findall(r'\(\s*\\[a-zA-Z][^)]{0,80}\)', resp)
        if malformed:
            print(f"\nMalformed '( \\cmd )' patterns: {len(malformed)}")
            for m in malformed[:5]:
                print(f"  {repr(m[:60])}")


def test_render_pattern(thread_id: str, exchange_idx: int = 3):
    """Test what the renderLLMResponse regex would match."""
    path = EXPLORATIONS_DIR / f"{thread_id}.json"
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    resp = data['exchanges'][exchange_idx]['response']

    # The JavaScript pattern: /\(\s*(\\[a-zA-Z]+[^)]*)\s*\)/g
    pattern = r'\(\s*(\\[a-zA-Z]+[^)]*)\s*\)'

    matches = list(re.finditer(pattern, resp))
    print(f"Pattern matches: {len(matches)}")
    for i, m in enumerate(matches[:10]):
        print(f"{i+1}. {repr(m.group(0)[:80])}")


def show_sample(thread_id: str, exchange_idx: int, start: int, length: int = 200):
    """Show a sample of text from an exchange."""
    path = EXPLORATIONS_DIR / f"{thread_id}.json"
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    resp = data['exchanges'][exchange_idx]['response']
    sample = resp[start:start+length]
    print(f"Sample from exchange {exchange_idx}, chars {start}-{start+length}:")
    print(repr(sample))
    print()
    print("Rendered:")
    print(sample)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python debug_util.py analyze <thread_id> [exchange_idx]")
        print("  python debug_util.py test <thread_id> [exchange_idx]")
        print("  python debug_util.py sample <thread_id> <exchange_idx> <start> [length]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "analyze":
        thread_id = sys.argv[2] if len(sys.argv) > 2 else "084cf1c9-a6c"
        ex_idx = int(sys.argv[3]) if len(sys.argv) > 3 else None
        analyze_math_patterns(thread_id, ex_idx)

    elif cmd == "test":
        thread_id = sys.argv[2] if len(sys.argv) > 2 else "084cf1c9-a6c"
        ex_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        test_render_pattern(thread_id, ex_idx)

    elif cmd == "sample":
        thread_id = sys.argv[2]
        ex_idx = int(sys.argv[3])
        start = int(sys.argv[4])
        length = int(sys.argv[5]) if len(sys.argv) > 5 else 200
        show_sample(thread_id, ex_idx, start, length)
