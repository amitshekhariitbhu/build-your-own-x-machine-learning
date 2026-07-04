import numpy as np
import re, zlib

# Text extraction from PDFs, from scratch.
#
# A PDF is a tree of numbered "objects"; page text lives in *content streams*
# (usually FlateDecode/zlib compressed) as a tiny drawing language:
#     BT  /F1 12 Tf  72 740 Td  (Hello) Tj  ...  ET
# To recover the text we must (1) split the file into objects, (2) follow the
# /Pages -> /Kids -> /Contents references in reading order, (3) inflate each
# content stream, and (4) parse its operators, pulling the literal `(...)`,
# hex `<...>` and kerned `[..]TJ` strings shown by Tj/TJ.  A naive "grep the
# printable bytes" reader sees only compressed garbage -- our baseline here.
#
# We first BUILD a valid synthetic PDF (planted, known ground-truth lines) so
# the whole thing is self-contained, then EXTRACT and score recovery vs truth.


# --------------------------------------------------------------- PDF writer
def esc(s):                                    # escape literal-string metachars
    return s.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)').encode('latin-1')


def make_content(lines):
    # Emit one text-showing operator per line, cycling Tj / kerned-TJ / hex
    # so the parser must handle all three encodings.  Td triggers line breaks.
    out, y = [b'BT', b'/F1 12 Tf'], 740
    for i, line in enumerate(lines):
        if i % 3 == 0:                                       # literal (...) Tj
            out.append(b'72 %d Td (%s) Tj' % (y, esc(line)))
        elif i % 3 == 1:                                     # kerned [..] TJ
            m = max(1, len(line) // 2)
            arr = b'(%s) -40 (%s)' % (esc(line[:m]), esc(line[m:]))
            out.append(b'72 %d Td [ %s ] TJ' % (y, arr))
        else:                                                # hex <..> Tj
            out.append(b'72 %d Td <%s> Tj' % (y, line.encode('latin-1').hex().encode()))
        y -= 16
    out.append(b'ET')
    return b'\n'.join(out)


def build_pdf(pages):
    parts, page_nums, nxt = {}, [], 4
    for lines in pages:
        cnum, pnum, nxt = nxt, nxt + 1, nxt + 2
        comp = zlib.compress(make_content(lines))
        parts[cnum] = (b'<< /Length %d /Filter /FlateDecode >>\nstream\n' % len(comp)
                       + comp + b'\nendstream')
        parts[pnum] = (b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] '
                       b'/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>' % cnum)
        page_nums.append(pnum)
    kids = b' '.join(b'%d 0 R' % p for p in page_nums)
    parts[1] = b'<< /Type /Catalog /Pages 2 0 R >>'
    parts[2] = b'<< /Type /Pages /Kids [ %s ] /Count %d >>' % (kids, len(page_nums))
    parts[3] = b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'

    body, offs = b'%PDF-1.4\n', {}
    for num in range(1, nxt):
        offs[num] = len(body)
        body += b'%d 0 obj\n' % num + parts[num] + b'\nendobj\n'
    xref_pos, n = len(body), nxt
    xref = b'xref\n0 %d\n0000000000 65535 f \n' % n
    for num in range(1, n):
        xref += b'%010d 00000 n \n' % offs[num]
    return body + xref + (b'trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF'
                          % (n, xref_pos))


# --------------------------------------------------------------- PDF parser
def load_objects(pdf):
    return {int(m.group(1)): m.group(2)
            for m in re.finditer(rb'(\d+)\s+\d+\s+obj(.*?)endobj', pdf, re.DOTALL)}


def stream_bytes(body):
    s = body.find(b'stream')
    if s < 0:
        return None
    j = s + 6
    j += 2 if body[j:j + 2] == b'\r\n' else 1        # skip EOL after 'stream'
    m = re.search(rb'/Length\s+(\d+)', body)
    data = body[j:j + int(m.group(1))] if m else body[j:body.find(b'endstream', j)]
    return zlib.decompress(data) if b'/FlateDecode' in body else data


def read_literal(d, i, n):                            # parse (...) with escapes
    depth, out = 1, []
    esc_map = {0x6E: '\n', 0x72: '\r', 0x74: '\t', 0x62: '\b', 0x66: '\f',
               0x28: '(', 0x29: ')', 0x5C: '\\'}
    while i < n:
        c = d[i]
        if c == 0x5C:                                 # backslash escape
            i += 1
            e = d[i]
            if e in esc_map:
                out.append(esc_map[e]); i += 1
            elif 0x30 <= e <= 0x37:                   # \ddd octal
                oc = ''
                while i < n and len(oc) < 3 and 0x30 <= d[i] <= 0x37:
                    oc += chr(d[i]); i += 1
                out.append(chr(int(oc, 8)))
            elif e in (0x0A, 0x0D):                    # line continuation
                i += 1
            else:
                out.append(chr(e)); i += 1
        elif c == 0x28:
            depth += 1; out.append('('); i += 1
        elif c == 0x29:
            depth -= 1; i += 1
            if depth == 0:
                break
            out.append(')')
        else:
            out.append(chr(c)); i += 1
    return ''.join(out), i


def read_hex(d, i, n):                                 # parse <48656C..>
    h = ''
    while i < n and d[i] != 0x3E:
        if chr(d[i]) in '0123456789abcdefABCDEF':
            h += chr(d[i])
        i += 1
    if len(h) % 2:
        h += '0'
    return ''.join(chr(int(h[k:k + 2], 16)) for k in range(0, len(h), 2)), i + 1


def tokenize(d):
    toks, i, n = [], 0, len(d)
    while i < n:
        c = d[i]
        if c == 0x28:                                 # (
            s, i = read_literal(d, i + 1, n); toks.append(('str', s))
        elif c == 0x3C:                               # <  (content has no dicts)
            s, i = read_hex(d, i + 1, n); toks.append(('str', s))
        elif c in b' \t\r\n[]':
            i += 1
        else:                                         # operator / number / name
            j = i
            while j < n and d[j] not in b' \t\r\n()<>[]/':
                j += 1
            j = j + 1 if j == i else j
            toks.append(('op', d[i:j].decode('latin-1'))); i = j
    return toks


def toks_to_text(toks):
    lines, cur, pending = [], [], []
    for kind, val in toks:
        if kind == 'str':
            pending.append(val)
        elif val in ('Tj', 'TJ'):                     # show buffered strings
            cur.extend(pending); pending = []
        elif val in ('Td', 'TD', 'T*', 'ET'):         # move -> line break
            if cur:
                lines.append(''.join(cur)); cur = []
    if cur:
        lines.append(''.join(cur))
    return '\n'.join(lines)


def extract_text(pdf):
    objs = load_objects(pdf)
    pages_body = next(b for b in objs.values() if b'/Kids' in b and b'/Pages' in b)
    order = re.findall(rb'(\d+)\s+0\s+R',
                       re.search(rb'/Kids\s*\[(.*?)\]', pages_body).group(1))
    texts = []
    for pnum in (int(x) for x in order):
        cm = re.search(rb'/Contents\s+(\d+)\s+0\s+R', objs.get(pnum, b''))
        if cm:
            texts.append(toks_to_text(tokenize(stream_bytes(objs[int(cm.group(1))]))))
    return '\n'.join(texts)


def naive_extract(pdf):                                # baseline: printable bytes
    return ' '.join(r.decode('latin-1') for r in re.findall(rb'[\x20-\x7e]{3,}', pdf))


# --------------------------------------------------------------- scoring
def word_recall(pred, truth):
    from collections import Counter
    pc, tc = Counter(pred.split()), Counter(truth.split())
    return sum(min(pc[w], tc[w]) for w in tc), sum(tc.values())


if __name__ == "__main__":
    np.random.seed(0)
    SUBJ = ["The invoice", "Our report", "This memo", "Account 42", "The ledger"]
    VERB = ["lists", "shows", "records", "confirms", "details", "reports"]
    OBJ = ["the totals", "a refund", "net revenue", "the taxes", "the fees"]
    TAIL = ["for March.", "this quarter.", "in full.", "as agreed.", "on file."]
    special = ["Payment (net) done \\ ok", "Ref: item(2) & note\\path"]

    # Planted ground truth: several pages of templated, known lines.
    pages = []
    for _ in range(4):
        lines = [" ".join([np.random.choice(SUBJ), np.random.choice(VERB),
                            np.random.choice(OBJ), np.random.choice(TAIL)])
                 for _ in range(5)]
        lines.append(np.random.choice(special))          # escape-sensitive line
        pages.append(lines)

    pdf = build_pdf(pages)
    truth = "\n".join("\n".join(p) for p in pages)

    got = extract_text(pdf)
    base = naive_extract(pdf)
    g_hit, tot = word_recall(got, truth)
    b_hit, _ = word_recall(base, truth)
    exact = got == truth

    print("Synthetic PDF: %d bytes, %d pages, %d words (FlateDecode compressed)"
          % (len(pdf), len(pages), tot))
    print("-" * 60)
    print("From-scratch parser  word recall: %.4f  (%d/%d)" % (g_hit / tot, g_hit, tot))
    print("Naive byte-grep base word recall: %.4f  (%d/%d)" % (b_hit / tot, b_hit, tot))
    print("Exact full-text reconstruction  : %s" % exact)
    print("-" * 60)
    print("First recovered line: %r" % got.splitlines()[0])
    print("Escape-handled line : %r" % [l for l in got.splitlines() if '(' in l][0])
    print("-" * 60)
    print("Parser beats naive baseline: %s" % (g_hit / tot > 0.9 and b_hit < g_hit))
