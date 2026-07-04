import numpy as np

# Build a Barcode and QR Code Reader System from scratch.
# Two decoders, both driven only by raw pixel matrices we synthesise here:
#   (1) A 1-D BARCODE reader in the EAN-13 spirit. Each digit -> a fixed 7-module
#       black/white pattern (the real EAN "L" table). We render the bars to a
#       noisy, blurred pixel image at an UNKNOWN scale, then the reader recovers
#       the digits: collapse rows -> find the code span -> estimate module width
#       from the known layout -> sample each module centre -> nearest-pattern
#       match (which also corrects a stray flipped module).
#   (2) A 2-D QR-LIKE matrix reader. A message is Hamming(7,4)-encoded (so each
#       block survives 1 bit error), laid into a grid, XOR-masked, and stamped
#       with ring finder patterns in 3 corners (4th left blank). The image is
#       RANDOMLY ROTATED by a multiple of 90 deg and its data modules are flipped
#       with probability p. The reader detects orientation from the finders,
#       unmasks, and Hamming-corrects the payload back out.
# Correctness signals (over many random codes): barcode per-digit accuracy and
# exact-code rate vs a 10%/digit random guess; QR char-accuracy WITH error
# correction vs WITHOUT and vs a 1/256 random baseline, under rotation + noise.

# ---- EAN-13 left-hand "L" digit table: 7 modules each (1 = black bar) --------
EAN_L = ["0001101", "0011001", "0010011", "0111101", "0100011",
         "0110001", "0101111", "0111011", "0110111", "0001011"]
PAT = np.array([[int(b) for b in s] for s in EAN_L])      # (10, 7)
GUARD = np.array([1, 0, 1])                               # start / end guard


class BarcodeReader:
    """Render digits to a pixel image and decode them back (EAN-13 style)."""

    def encode(self, digits, module_px=4, rows=8):
        mods = np.concatenate([GUARD] + [PAT[d] for d in digits] + [GUARD])
        line = np.repeat(mods.astype(float), module_px)          # modules -> pixels
        img = np.tile(line, (rows, 1))
        return img, len(digits)

    def _blur_noise(self, img, rng, sigma=0.20):
        k = np.ones(3) / 3.0                                     # mild optical blur
        img = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, img)
        return img + rng.randn(*img.shape) * sigma

    def decode(self, img, n_digits):
        sig = img.mean(axis=0)                                   # collapse rows
        black = sig > 0.5
        xs = np.where(black)[0]
        lo, hi = xs[0], xs[-1]                                   # code span
        n_mod = 7 * n_digits + 6                                 # +2 guards (3 each)
        w = (hi - lo + 1) / n_mod                                # est. module width
        vals = np.array([sig[int(lo + (m + 0.5) * w)] for m in range(n_mod)])
        bits = (vals > 0.5).astype(int)[3:3 + 7 * n_digits]      # drop start guard
        out = []
        for i in range(n_digits):
            chunk = bits[i * 7:(i + 1) * 7]
            out.append(int(np.abs(PAT - chunk).sum(axis=1).argmin()))  # nearest pattern
        return out


# ---- Hamming(7,4): single-error-correcting code over GF(2) -------------------
G = np.array([[1, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 1],
              [0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]])          # data -> code
H = np.array([[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0],
              [0, 1, 1, 1, 0, 0, 1]])                                  # parity check


def hamming_encode(bits):
    bits = np.asarray(bits).reshape(-1, 4)
    return ((bits @ G) % 2).reshape(-1)


def hamming_decode(code, correct=True):
    C = np.asarray(code).reshape(-1, 7).copy()
    if correct:
        for row in C:                                            # fix 1 error / block
            s = (H @ row) % 2
            if s.any():
                j = np.where((H.T == s).all(axis=1))[0]
                if len(j):
                    row[j[0]] ^= 1
    return C[:, :4].reshape(-1)


def bytes_to_bits(bs):
    return np.array([(b >> k) & 1 for b in bs for k in range(7, -1, -1)])


def bits_to_bytes(bits):
    bits = np.asarray(bits).reshape(-1, 8)
    return [int("".join(map(str, r)), 2) for r in bits]


class QRCodeReader:
    """Ring finders + Hamming payload; reads through rotation and module noise."""

    def __init__(self, N=13):
        self.N = N
        self.ring = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])  # corner finder

    def _cells(self):                                            # data cells, row-major
        N = self.N
        corner = lambda r, c: (r < 3 or r >= N - 3) and (c < 3 or c >= N - 3)
        return [(r, c) for r in range(N) for c in range(N) if not corner(r, c)]

    def encode(self, message):
        N = self.N
        bits = hamming_encode(bytes_to_bits([ord(ch) for ch in message]))
        g = np.zeros((N, N), int)
        g[0:3, 0:3] = g[0:3, N - 3:N] = g[N - 3:N, 0:3] = self.ring   # 3 finders
        for i, (r, c) in enumerate(self._cells()):
            b = bits[i] if i < len(bits) else 0
            g[r, c] = b ^ ((r + c) & 1)                          # XOR mask
        return g

    def _orient(self, g):
        # Rotate so the blank corner (no finder) sits bottom-right.
        best, bg = -1, g
        for k in range(4):
            c = np.rot90(g, k)
            s = c[:3, :3].sum() + c[:3, -3:].sum() + c[-3:, :3].sum() - c[-3:, -3:].sum()
            if s > best:
                best, bg = s, c
        return bg

    def decode(self, g, n_bytes, correct=True):
        g = self._orient(g)
        bits = [g[r, c] ^ ((r + c) & 1) for (r, c) in self._cells()]
        code = np.array(bits[:7 * 2 * n_bytes])                  # 2 nibbles / byte
        data = hamming_decode(code, correct)
        return "".join(chr(b) for b in bits_to_bytes(data))


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    # ---------------- 1-D barcode benchmark ----------------
    bc = BarcodeReader()
    D, trials = 8, 300
    dig_hits = code_hits = total = 0
    for _ in range(trials):
        digits = rng.randint(0, 10, D)
        img, n = bc.encode(digits)
        img = bc._blur_noise(img, rng)
        pred = bc.decode(img, n)
        dig_hits += int((np.array(pred) == digits).sum())
        code_hits += int(np.array_equal(pred, digits))
        total += D
    bc_dig = dig_hits / total
    bc_code = code_hits / trials

    # ---------------- 2-D QR benchmark (rotation + noise) ----------------
    qr = QRCodeReader(N=13)
    alphabet = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
    L, qtrials, p = 6, 300, 0.05
    ec_chars = raw_chars = ec_exact = tot_chars = 0
    for _ in range(qtrials):
        msg = "".join(rng.choice(alphabet, L))
        g = qr.encode(msg)
        g = np.rot90(g, rng.randint(0, 4))                       # unknown orientation
        cells = qr._cells()                                      # noise on data only
        for (r, c) in cells:
            if rng.rand() < p:
                g[r, c] ^= 1
        dec_ec = qr.decode(g.copy(), L, correct=True)
        dec_raw = qr.decode(g.copy(), L, correct=False)
        ec_chars += sum(a == b for a, b in zip(dec_ec, msg))
        raw_chars += sum(a == b for a, b in zip(dec_raw, msg))
        ec_exact += int(dec_ec == msg)
        tot_chars += L
    qr_ec = ec_chars / tot_chars
    qr_raw = raw_chars / tot_chars
    qr_exact = ec_exact / qtrials

    print("=== 1-D Barcode reader (EAN-13 style, blur + noise) ===")
    print("Codes: %d   Digits/code: %d" % (trials, D))
    print("Per-digit accuracy   reader : %.3f" % bc_dig)
    print("Per-digit accuracy   random : %.3f  (1/10)" % 0.10)
    print("Exact-code recovery  reader : %.3f" % bc_code)
    print("Exact-code recovery  random : %.3f  (0.1^%d)" % (0.1 ** D, D))
    print("-" * 60)
    print("=== 2-D QR-like reader (rotation + %.0f%% module noise) ===" % (p * 100))
    print("Codes: %d   Chars/code: %d" % (qtrials, L))
    print("Char accuracy  WITH  Hamming EC : %.3f" % qr_ec)
    print("Char accuracy  WITHOUT     EC   : %.3f" % qr_raw)
    print("Char accuracy  random           : %.4f  (1/256)" % (1 / 256))
    print("Exact-message recovery (EC)     : %.3f" % qr_exact)
    print("-" * 60)
    beats = (bc_dig > 0.9 and bc_code > 0.5 and qr_ec > qr_raw > 1 / 256)
    print("Both readers beat their baselines:", beats)
