import ast
import re


class DiffApp:

    def __init__(self, success, code=None, error=None, is_full_replacement=False):
        self.success = success
        self.code = code
        self.error = error
        self.is_full_replacement = is_full_replacement

    def valid_python(self):
        if not self.code:
            return False
        try:
            ast.parse(self.code)
            return True
        except SyntaxError:
            return False


def normalize_nl(text):
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def f_nl(s):
    return s if s.endswith('\n') else s + '\n'


def find_match(source_lines, search_pattern, start_hint=None):
    """Find where search_pattern appears in source_lines, ignoring leading/trailing whitespace and blank lines.
    
    Returns (start_pos, end_pos) tuple or None if not found.
    end_pos is exclusive (like slice notation).
    """
    if not search_pattern:
        return None

    total_lines = len(source_lines)
    pattern_len = len(search_pattern)
    if pattern_len > total_lines:
        return None

    stripped_source = [line.strip() for line in source_lines]
    stripped_pattern = [line.strip() for line in search_pattern]
    
    # Filter out empty lines from pattern for matching
    non_empty_pattern = [l for l in stripped_pattern if l]
    if not non_empty_pattern:
        return None

    positions_to_check = []
    if start_hint is not None and 0 <= start_hint < total_lines:
        positions_to_check.append(start_hint)
        for offset in range(1, 11):
            if start_hint - offset >= 0:
                positions_to_check.append(start_hint - offset)
            if start_hint + offset < total_lines:
                positions_to_check.append(start_hint + offset)
    positions_to_check.extend(range(total_lines))

    checked = set()
    for pos in positions_to_check:
        if pos in checked:
            continue
        checked.add(pos)
        
        # Try to match non-empty pattern lines against source, skipping empty source lines
        src_idx = pos
        pat_idx = 0
        
        while pat_idx < len(non_empty_pattern) and src_idx < total_lines:
            if not stripped_source[src_idx]:  # skip empty source lines
                src_idx += 1
                continue
            if stripped_source[src_idx] == non_empty_pattern[pat_idx]:
                pat_idx += 1
                src_idx += 1
            else:
                break
        
        if pat_idx == len(non_empty_pattern):
            return (pos, src_idx)  # Return start and end positions

    return None


def parse_hunk(hunk_lines):
    # parse diff hunk lines -> removed_lines, added_lines
    removed, added = [], []
    has_context_prefix = any(line.startswith(" ") for line in hunk_lines)

    for line in hunk_lines:
        if line.startswith("-"):
            removed.append(f_nl(line[1:]))
        elif line.startswith("+"):
            added.append(f_nl(line[1:]))
        elif line.startswith(" "):
            context = f_nl(line[1:])
            removed.append(context)
            added.append(context)
        elif not has_context_prefix and not line.startswith(("-", "+")):
            # bare context line (no prefix)
            context = f_nl(line)
            removed.append(context)
            added.append(context)

    return removed, added


def extract_hunks(diff_text):
    """Extract hunks from diff text. Returns [(removed, added, line_number), ...]"""
    lines = diff_text.splitlines()
    
    # Strip markdown code fences
    lines = [line for line in lines if not line.strip().startswith('```')]

    # check for V4A format markers
    body_lines = []
    in_v4a = False
    for line in lines:
        if "*** Begin Patch" in line or "*** Update File" in line:
            in_v4a = True
            continue
        if "*** End Patch" in line:
            break
        if in_v4a:
            body_lines.append(line)

    # fall back -> filter diff header
    if not body_lines:
        skip_prefixes = ("diff --git", "index ", "--- ", "+++ ", "similarity")
        body_lines = [line for line in lines if not line.startswith(skip_prefixes)]

    # split into @@ hunks
    hunks = []
    current_hunk = []
    expected_line = None

    for line in body_lines:
        if line.startswith("@@") or line.strip() == "@@":
            if current_hunk:
                removed, added = parse_hunk(current_hunk)
                hunks.append((removed, added, expected_line))
            current_hunk = []
            match = re.match(r'@@\s*-(\d+)', line)
            expected_line = int(match.group(1)) - 1 if match else None
        else:
            current_hunk.append(line)

    if current_hunk:
        removed, added = parse_hunk(current_hunk)
        hunks.append((removed, added, expected_line))

    return hunks


def apply_diff(original, diff, require_valid_python=True):
    ## apply diff to og mutation
    original = normalize_nl(original)
    diff = normalize_nl(diff).strip()

    if not diff:
        return DiffApp(False, error="Empty diff")

    hunks = extract_hunks(diff)
    if not hunks:
        return DiffApp(False, error="No hunks found")

    # full replacement (pure addition hunks)
    if all(not removed for removed, added, _ in hunks):
        code = ''.join(line for _, added, _ in hunks for line in added)
        return DiffApp(True, code, is_full_replacement=True)

    # each hunk
    source_lines = original.splitlines(keepends=True)
    if source_lines and not source_lines[-1].endswith('\n'):
        source_lines[-1] += '\n'

    line_offset = 0
    for removed, added, expected_at in hunks:
        if not removed:
            continue

        adj_exp = expected_at + line_offset if expected_at is not None else None
        match_result = find_match(source_lines, removed, adj_exp)

        if match_result is None:
            snippet = removed[0][:40].strip() if removed else "?"
            return DiffApp(False, error=f"Context not found: '{snippet}...'")

        start_pos, end_pos = match_result
        matched_len = end_pos - start_pos
        source_lines[start_pos:end_pos] = added
        line_offset += len(added) - matched_len

    result = DiffApp(True, ''.join(source_lines))

    if require_valid_python and not result.valid_python():
        return DiffApp(False, error="Result is not valid Python")

    return result
