Codex made me a simple and self-contained Reed-Solomon FEC codec that stores our 105 bytes of payload into 256-byte blocks. The payload is like above, 105 bytes max, zero-extended as needed. The payload block is then stored like:

```python
uint8     header        # =1 used as version field for Reed-Solomon codec
byte[105] data          # Protected useful payload zero-padded to the fixed size.
uint32    crc32         # CRC32 IEEE of the data block
byte[146] reed_solomon  # RS(255,109) forward error correction (FEC) data
```

[details="rs255_109_cli.c"]
```c
/*
  rs255_109_cli.c

  Fixed record size: 256 bytes on storage:
    [0]      = 1-byte header/version (0x01)
    [1..255] = 255-byte RS(255,109) codeword over GF(256), interleaved by permutation.

  RS payload (systematic):
    data[0..104]  = user data (105 bytes, zero-extended if input shorter)
    data[105..108]= CRC32 (IEEE/PKZIP, polynomial 0xEDB88320), little-endian
    parity[0..145]= RS parity (146 bytes)
    codeword[0..254] = data||parity

  Interleaver:
    stored[j] = codeword[i] where j = (i * STRIDE) mod 255, STRIDE coprime with 255 (here 29).
    Decoder uses modular inverse to deinterleave.

  Usage:
    Encode:
      ./rs255_109_cli enc <hexdata> > record.hex
        - hexdata may be shorter than 105 bytes (210 hex chars); it is zero-extended to 105.
        - If longer than 105 bytes, extra bytes are ignored.
        - Emits 256 record bytes as 512 lowercase hex chars plus newline.

    Decode:
      ./rs255_109_cli dec < record.hex > data.hex
        - Reads stdin as hex text (whitespace ignored), expecting exactly 256 decoded bytes.
        - Also accepts legacy raw 256-byte binary records.
        - Outputs 105 bytes as lowercase hex to stdout if CRC verifies.
        - Exits nonzero on failure.

  Notes:
    - This is a compact, self-contained RS(255,109) encoder/decoder.
    - It corrects up to t = floor(146/2) = 73 unknown byte errors.
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>

#define N 255
#define K 109
#define PAR (N - K)          // 146
#define USER_DATA 105
#define HDR 0x01

#define PRIM_POLY 0x11d      // x^8 + x^4 + x^3 + x^2 + 1
#define STRIDE 29

static uint8_t gf_exp[512];
static uint8_t gf_log[256];

static void gf_init(void) {
  uint16_t x = 1;
  for (int i = 0; i < 255; i++) {
    gf_exp[i] = (uint8_t)x;
    gf_log[(uint8_t)x] = (uint8_t)i;
    x <<= 1;
    if (x & 0x100) x ^= PRIM_POLY;
  }
  gf_log[0] = 0; // unused
  for (int i = 255; i < 512; i++) gf_exp[i] = gf_exp[i - 255];
}

static inline uint8_t gf_mul(uint8_t a, uint8_t b) {
  if (!a || !b) return 0;
  return gf_exp[gf_log[a] + gf_log[b]];
}

static inline uint8_t gf_div(uint8_t a, uint8_t b) {
  if (!a) return 0;
  if (!b) return 0; // invalid
  int t = (int)gf_log[a] - (int)gf_log[b];
  if (t < 0) t += 255;
  return gf_exp[t];
}

static inline uint8_t gf_pow(uint8_t a, int p) {
  if (p == 0) return 1;
  if (!a) return 0;
  int e = (gf_log[a] * p) % 255;
  if (e < 0) e += 255;
  return gf_exp[e];
}

static inline uint8_t gf_inv(uint8_t a) {
  // a^(254)
  if (!a) return 0;
  return gf_exp[255 - gf_log[a]];
}

// CRC32 (IEEE)
static uint32_t crc32_ieee(const uint8_t *data, size_t len) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; i++) {
    crc ^= (uint32_t)data[i];
    for (int b = 0; b < 8; b++) {  // TODO this is slow, use tables.
      uint32_t mask = -(crc & 1u);
      crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
  }
  return ~crc;
}

// Compute generator polynomial g(x) = Π_{i=0..PAR-1} (x - α^i).
// Coefficients are stored in descending order: g[0]x^PAR + ... + g[PAR].
static void rs_generator(uint8_t *g /* PAR+1 */) {
  memset(g, 0, PAR + 1);
  g[0] = 1;
  int deg = 0;
  for (int i = 0; i < PAR; i++) {
    uint8_t a = gf_exp[i];
    uint8_t next[PAR + 1];
    memset(next, 0, sizeof(next));
    for (int j = 0; j <= deg; j++) {
      next[j] ^= g[j];
      next[j + 1] ^= gf_mul(g[j], a);
    }
    deg++;
    memcpy(g, next, PAR + 1);
  }
}

// Systematic encode: parity = remainder of (data * x^PAR) mod g(x).
static void rs_encode(const uint8_t *data_k, uint8_t *parity /* PAR */) {
  static uint8_t g[PAR + 1];
  static int inited = 0;
  if (!inited) { rs_generator(g); inited = 1; }

  uint8_t work[N];
  memset(work, 0, sizeof(work));
  memcpy(work, data_k, K);

  for (int i = 0; i < K; i++) {
    uint8_t coef = work[i];
    if (!coef) continue;
    for (int j = 1; j <= PAR; j++) {
      work[i + j] ^= gf_mul(coef, g[j]);
    }
  }

  memcpy(parity, work + K, PAR);
}

// Evaluate polynomial poly (degree <= deg) at x, coefficients ascending
static uint8_t poly_eval(const uint8_t *poly, int deg, uint8_t x) {
  uint8_t y = poly[deg];
  for (int i = deg - 1; i >= 0; i--) {
    y = gf_mul(y, x) ^ poly[i];
  }
  return y;
}

// Compute syndromes S_i = c(α^i), i=0..PAR-1; returns 1 if any nonzero
static int rs_syndromes(const uint8_t *cw /* N */, uint8_t *S /* PAR */) {
  int nonzero = 0;
  for (int i = 0; i < PAR; i++) {
    uint8_t x = gf_exp[i];
    uint8_t s = 0;
    // Horner over codeword as polynomial with cw[0] as highest degree or lowest?
    // Here we treat cw as polynomial with cw[0] at degree N-1 (standard for RS codes).
    // Evaluate: s = Σ cw[j] * x^(N-1-j)
    for (int j = 0; j < N; j++) {
      s = gf_mul(s, x) ^ cw[j];
    }
    S[i] = s;
    if (s) nonzero = 1;
  }
  return nonzero;
}

// Berlekamp–Massey to find error locator sigma(x) from syndromes
// sigma[0]=1, degree L returned
static int berlekamp_massey(const uint8_t *S, uint8_t *sigma /* PAR+1 */) {
  uint8_t C[PAR + 1], B[PAR + 1];
  memset(C, 0, sizeof(C));
  memset(B, 0, sizeof(B));
  C[0] = 1; B[0] = 1;

  int L = 0;
  int m = 1;
  uint8_t b = 1;

  for (int n = 0; n < PAR; n++) {
    // discrepancy d = S[n] + Σ_{i=1..L} C[i]*S[n-i]
    uint8_t d = S[n];
    for (int i = 1; i <= L; i++) {
      d ^= gf_mul(C[i], S[n - i]);
    }
    if (!d) {
      m++;
      continue;
    }
    uint8_t T[PAR + 1];
    memcpy(T, C, sizeof(T));

    uint8_t coef = gf_div(d, b);
    // C(x) = C(x) - coef * x^m * B(x)  (minus == plus)
    for (int i = 0; i + m <= PAR; i++) {
      C[i + m] ^= gf_mul(coef, B[i]);
    }
    if (2 * L <= n) {
      L = n + 1 - L;
      memcpy(B, T, sizeof(B));
      b = d;
      m = 1;
    } else {
      m++;
    }
  }
  memcpy(sigma, C, PAR + 1);
  return L;
}

// Compute error evaluator omega(x) = (S(x)*sigma(x)) mod x^PAR
// syndromes polynomial S(x) = S0 + S1 x + ...
static void compute_omega(const uint8_t *S, const uint8_t *sigma, uint8_t *omega /* PAR */) {
  uint8_t tmp[2 * PAR + 1];
  memset(tmp, 0, sizeof(tmp));
  for (int i = 0; i <= PAR; i++) {
    if (!sigma[i]) continue;
    for (int j = 0; j < PAR; j++) {
      if (!S[j]) continue;
      tmp[i + j] ^= gf_mul(sigma[i], S[j]);
    }
  }
  // mod x^PAR => take lower PAR terms
  for (int i = 0; i < PAR; i++) omega[i] = tmp[i];
}

// Formal derivative of sigma: sigma'(x) (only odd powers survive in GF(2))
static int sigma_derivative(const uint8_t *sigma, int deg, uint8_t *ds /* PAR */) {
  memset(ds, 0, PAR);
  int ddeg = 0;
  for (int i = 1; i <= deg; i++) {
    if (i & 1) { // odd
      ds[i - 1] = sigma[i];
      ddeg = i - 1;
    }
  }
  return ddeg;
}

// Chien search: find roots of sigma(x). Returns number of errors and their positions (0..N-1)
// Using convention: root at x = α^{-pos} where pos corresponds to codeword index.
static int chien_search(const uint8_t *sigma, int deg, int *pos /* PAR */) {
  int count = 0;
  for (int i = 0; i < N; i++) {
    // x = α^{i}
    // With our syndrome convention S_i = sum(e_p * X_p^i), X_p = α^{N-1-p},
    // sigma roots are at X_p^{-1} = α^{p+1}, so p = i - 1 mod N.
    uint8_t x = gf_exp[i];
    uint8_t y = poly_eval(sigma, deg, x);
    if (y == 0) {
      int p = (i + (N - 1)) % N;
      if (p < 0 || p >= N) continue;
      pos[count++] = p;
      if (count >= PAR) break;
    }
  }
  return count;
}

// Forney algorithm: compute error magnitudes and correct codeword
// Returns 0 on success, nonzero on failure
static int rs_correct(uint8_t *cw /* N */, const uint8_t *S, const uint8_t *sigma, int sig_deg, const int *pos, int npos) {
  uint8_t omega[PAR];
  compute_omega(S, sigma, omega);

  uint8_t ds[PAR];
  int ds_deg = sigma_derivative(sigma, sig_deg, ds);

  for (int i = 0; i < npos; i++) {
    int p = pos[i]; // 0..N-1
    // X_p^{-1} = α^{p+1}, X_p = α^{255-(p+1)}.
    int xinv_exp = (p + 1) % 255;
    uint8_t Xinv = gf_exp[xinv_exp];
    uint8_t X = gf_exp[(255 - xinv_exp) % 255];

    uint8_t num = poly_eval(omega, PAR - 1, Xinv);
    uint8_t den = poly_eval(ds, ds_deg, Xinv);
    if (!den) return 1;

    // For S_0-based syndromes, Forney requires an extra X factor.
    uint8_t mag = gf_mul(X, gf_div(num, den));
    // minus == plus
    cw[p] ^= mag;
  }
  return 0;
}

// RS decode (errors-only). Returns 0 on success (syndromes cleared), nonzero on failure.
static int rs_decode(uint8_t *cw /* N */) {
  uint8_t S[PAR];
  if (!rs_syndromes(cw, S)) return 0; // already clean

  uint8_t sigma[PAR + 1];
  int L = berlekamp_massey(S, sigma);
  if (L <= 0 || L > PAR) return 1;

  int pos[PAR];
  int npos = chien_search(sigma, L, pos);
  if (npos != L) return 2; // couldn't find all roots

  if (rs_correct(cw, S, sigma, L, pos, npos)) return 3;

  // verify
  if (rs_syndromes(cw, S)) return 4;
  return 0;
}

// Extended Euclid for modular inverse mod 255 (for interleaver inverse)
static int modinv_255(int a) {
  int t = 0, newt = 1;
  int r = 255, newr = a % 255;
  while (newr != 0) {
    int q = r / newr;
    int tmp = t - q * newt; t = newt; newt = tmp;
    tmp = r - q * newr; r = newr; newr = tmp;
  }
  if (r > 1) return -1;
  if (t < 0) t += 255;
  return t;
}

static void interleave_255(const uint8_t *in, uint8_t *out) {
  for (int i = 0; i < 255; i++) {
    int j = (i * STRIDE) % 255;
    out[j] = in[i];
  }
}

static void deinterleave_255(const uint8_t *in, uint8_t *out) {
  static int inv = -2;
  if (inv == -2) inv = modinv_255(STRIDE);
  for (int j = 0; j < 255; j++) {
    int i = (j * inv) % 255;
    out[i] = in[j];
  }
}

static int hexval(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static size_t parse_hex_bytes(const char *hex, uint8_t *out, size_t max_out) {
  size_t n = 0;
  size_t len = strlen(hex);
  size_t i = 0;
  // skip optional 0x
  if (len >= 2 && hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) { hex += 2; len -= 2; }
  // ignore whitespace
  char *buf = (char*)malloc(len + 1);
  size_t bl = 0;
  for (size_t k = 0; k < len; k++) {
    if (!isspace((unsigned char)hex[k])) buf[bl++] = hex[k];
  }
  buf[bl] = 0;
  if (bl & 1) { free(buf); return 0; } // require even count
  for (i = 0; i + 1 < bl && n < max_out; i += 2) {
    int hi = hexval(buf[i]);
    int lo = hexval(buf[i + 1]);
    if (hi < 0 || lo < 0) { free(buf); return 0; }
    out[n++] = (uint8_t)((hi << 4) | lo);
  }
  free(buf);
  return n;
}

static void print_hex(const uint8_t *b, size_t n) {
  static const char *h = "0123456789abcdef";
  for (size_t i = 0; i < n; i++) {
    putchar(h[b[i] >> 4]);
    putchar(h[b[i] & 0xF]);
  }
  putchar('\n');
}

static int read_stdin_all(uint8_t **buf_out, size_t *len_out, int stop_at_newline) {
  size_t cap = 1024;
  size_t len = 0;
  uint8_t *buf = (uint8_t*)malloc(cap);
  if (!buf) return 1;

  int c;
  while ((c = fgetc(stdin)) != EOF) {
    if (stop_at_newline && c == '\n') break;
    if (len == cap) {
      size_t new_cap = cap * 2;
      if (new_cap < cap) {
        free(buf);
        return 1;
      }
      uint8_t *tmp = (uint8_t*)realloc(buf, new_cap);
      if (!tmp) {
        free(buf);
        return 1;
      }
      buf = tmp;
      cap = new_cap;
    }
    buf[len++] = (uint8_t)c;
  }
  if (ferror(stdin)) {
    free(buf);
    return 1;
  }

  *buf_out = buf;
  *len_out = len;
  return 0;
}

// Parse hex from a byte buffer. Sets *ok=1 for valid hex text and returns decoded byte count.
static size_t parse_hex_buffer_strict(const uint8_t *in, size_t in_len, uint8_t *out, size_t max_out, int *ok) {
  *ok = 0;
  size_t i = 0;
  while (i < in_len && isspace((unsigned char)in[i])) i++;
  if (i + 1 < in_len && in[i] == '0' && (in[i + 1] == 'x' || in[i + 1] == 'X')) i += 2;

  size_t n = 0;
  int hi = -1;
  for (; i < in_len; i++) {
    unsigned char c = in[i];
    if (isspace(c)) continue;
    int v = hexval(c);
    if (v < 0) return 0;
    if (hi < 0) {
      hi = v;
    } else {
      if (n >= max_out) return 0;
      out[n++] = (uint8_t)((hi << 4) | v);
      hi = -1;
    }
  }
  if (hi >= 0) return 0;
  *ok = 1;
  return n;
}

static int decode_record(const uint8_t *rec) {
  if (rec[0] != HDR) {
    fprintf(stderr, "bad header byte (expected 0x%02x, got 0x%02x)\n", HDR, rec[0]);
    return 3;
  }

  uint8_t stored255[255];
  memcpy(stored255, rec + 1, 255);

  uint8_t cw[N];
  deinterleave_255(stored255, cw);

  int rc = rs_decode(cw);
  if (rc != 0) {
    fprintf(stderr, "RS decode failed (%d)\n", rc);
    return 4;
  }

  // Extract and verify CRC
  uint8_t user[USER_DATA];
  memcpy(user, cw, USER_DATA);
  uint32_t crc_stored =
      ((uint32_t)cw[105]) |
      ((uint32_t)cw[106] << 8) |
      ((uint32_t)cw[107] << 16) |
      ((uint32_t)cw[108] << 24);
  uint32_t crc_calc = crc32_ieee(user, USER_DATA);
  if (crc_calc != crc_stored) {
    fprintf(stderr, "CRC mismatch (calc=0x%08x stored=0x%08x)\n", crc_calc, crc_stored);
    return 5;
  }

  // Output original 105 bytes as hex
  print_hex(user, USER_DATA);
  return 0;
}

static int do_encode(const char *hexarg) {
  uint8_t user[USER_DATA];
  memset(user, 0, sizeof(user));
  uint8_t tmp[USER_DATA];
  size_t got = parse_hex_bytes(hexarg, tmp, USER_DATA);
  if (got == 0 && strlen(hexarg) != 0) {
    fprintf(stderr, "hex parse error (need even number of hex digits)\n");
    return 2;
  }
  memcpy(user, tmp, got);

  // Build RS data (K bytes): 105 user bytes + CRC32(4)
  uint8_t data[K];
  memcpy(data, user, USER_DATA);
  uint32_t crc = crc32_ieee(user, USER_DATA);
  data[105] = (uint8_t)(crc & 0xFF);
  data[106] = (uint8_t)((crc >> 8) & 0xFF);
  data[107] = (uint8_t)((crc >> 16) & 0xFF);
  data[108] = (uint8_t)((crc >> 24) & 0xFF);

  uint8_t parity[PAR];
  rs_encode(data, parity);

  uint8_t cw[N];
  memcpy(cw, data, K);
  memcpy(cw + K, parity, PAR);

  // Interleave
  uint8_t stored255[255];
  interleave_255(cw, stored255);

  // Output 256 bytes: header + stored255
  uint8_t rec[256];
  rec[0] = HDR;
  memcpy(rec + 1, stored255, 255);

  print_hex(rec, sizeof(rec));
  return 0;
}

static int do_decode(const char *hexarg_opt) {
  uint8_t rec[256];
  if (hexarg_opt && hexarg_opt[0]) {
    int hex_ok = 0;
    size_t parsed = parse_hex_buffer_strict(
        (const uint8_t*)hexarg_opt, strlen(hexarg_opt), rec, sizeof(rec), &hex_ok);
    if (!hex_ok || parsed != sizeof(rec)) {
      fprintf(stderr, "dec argument must be exactly 256 bytes of hex (512 hex chars)\n");
      return 2;
    }
    return decode_record(rec);
  } else {
    uint8_t *in = NULL;
    size_t in_len = 0;
    int stop_at_newline = isatty(fileno(stdin)) ? 1 : 0;
    if (read_stdin_all(&in, &in_len, stop_at_newline)) {
      fprintf(stderr, "read error\n");
      return 2;
    }

    int hex_ok = 0;
    size_t parsed = parse_hex_buffer_strict(in, in_len, rec, sizeof(rec), &hex_ok);
    if (hex_ok) {
      if (parsed != sizeof(rec)) {
        fprintf(stderr, "hex input decoded to %zu bytes (expected 256)\n", parsed);
        free(in);
        return 2;
      }
    } else if (in_len == sizeof(rec)) {
      // Backward compatibility for older pipelines that pass raw binary records.
      memcpy(rec, in, sizeof(rec));
    } else {
      fprintf(stderr, "expected 512 hex chars (or 256 raw bytes), got %zu bytes\n", in_len);
      free(in);
      return 2;
    }
    free(in);
    return decode_record(rec);
  }
}

int main(int argc, char **argv) {
  gf_init();
  if (modinv_255(STRIDE) < 0) {
    fprintf(stderr, "configuration error: STRIDE=%d is not invertible modulo 255\n", STRIDE);
    return 2;
  }

  if (argc < 2) {
    fprintf(stderr, "usage:\n  %s enc <hexdata> > record.hex\n  %s dec [recordhex] < record.hex > data.hex\n", argv[0], argv[0]);
    return 2;
  }
  if (!strcmp(argv[1], "enc")) {
    if (argc < 3) {
      fprintf(stderr, "enc requires hex argument\n");
      return 2;
    }
    return do_encode(argv[2]);
  } else if (!strcmp(argv[1], "dec")) {
    return do_decode(argc >= 3 ? argv[2] : NULL);
  } else {
    fprintf(stderr, "unknown mode: %s (use enc/dec)\n", argv[1]);
    return 2;
  }
}
```
[/details]

[details="selftest_rs.py"]
```python
#!/usr/bin/env python3
"""
Randomized self-test for the educational RS(255,109) CLI.

Focus:
- Roundtrip sanity on clean records.
- Maximum theoretical random error correction: exactly t = 73 corrupted bytes
  in the RS-protected region (record bytes [1..255], header assumed intact).
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


USER_DATA = 105
RECORD_BYTES = 256
MAX_T = 73


@dataclass
class CmdResult:
    rc: int
    out: str
    err: str


def run_cmd(args: Sequence[str], cwd: Path) -> CmdResult:
    proc = subprocess.run(
        list(args),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return CmdResult(proc.returncode, proc.stdout.strip(), proc.stderr.strip())


def random_payload_hex(rng: random.Random) -> str:
    length = rng.randint(0, USER_DATA)
    try:
        payload = rng.randbytes(length)
    except AttributeError:
        payload = bytes(rng.getrandbits(8) for _ in range(length))
    return payload.hex()


def expected_decode_hex(input_hex: str) -> str:
    data = bytes.fromhex(input_hex)
    if len(data) > USER_DATA:
        data = data[:USER_DATA]
    return (data + b"\x00" * (USER_DATA - len(data))).hex()


def mutate_record_hex(
    record_hex: str, t: int, rng: random.Random
) -> Tuple[str, List[Tuple[int, int, int]]]:
    rec = bytearray.fromhex(record_hex)
    if len(rec) != RECORD_BYTES:
        raise ValueError(f"expected {RECORD_BYTES} bytes, got {len(rec)}")

    positions = rng.sample(range(1, RECORD_BYTES), t)
    mutations: List[Tuple[int, int, int]] = []
    for pos in positions:
        old = rec[pos]
        new = old
        while new == old:
            new = rng.randrange(0, 256)
        rec[pos] = new
        mutations.append((pos, old, new))
    mutations.sort(key=lambda x: x[0])
    return rec.hex(), mutations


def decode_should_match(cwd: Path, record_hex: str, expected_hex: str) -> Tuple[bool, CmdResult]:
    dec = run_cmd(["./a.out", "dec", record_hex], cwd)
    ok = dec.rc == 0 and dec.out == expected_hex
    return ok, dec


def compile_target(cwd: Path) -> None:
    r = run_cmd(["sh", "compile.sh"], cwd)
    if r.rc != 0:
        print("Compile failed", file=sys.stderr)
        if r.out:
            print(r.out, file=sys.stderr)
        if r.err:
            print(r.err, file=sys.stderr)
        raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Randomized RS self-test")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (printed and reused for repro)")
    parser.add_argument("--sanity-trials", type=int, default=50, help="Clean roundtrip trials")
    parser.add_argument("--trials", type=int, default=300, help="Random corruption trials at t=73")
    parser.add_argument("--t", type=int, default=MAX_T, help="Corruption count per trial (default 73)")
    args = parser.parse_args()

    if args.t < 0 or args.t > MAX_T:
        print(f"--t must be in [0, {MAX_T}]", file=sys.stderr)
        return 2

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 63)
    rng = random.Random(seed)
    cwd = Path(__file__).resolve().parent

    print(f"seed={seed}")
    print("compiling...")
    compile_target(cwd)

    # 1) Clean roundtrip sanity.
    print(f"sanity_trials={args.sanity_trials}")
    for i in range(args.sanity_trials):
        payload_hex = random_payload_hex(rng)
        exp = expected_decode_hex(payload_hex)

        enc = run_cmd(["./a.out", "enc", payload_hex], cwd)
        if enc.rc != 0:
            print(f"FAIL sanity[{i}] encode rc={enc.rc}", file=sys.stderr)
            print(f"payload_hex={payload_hex}", file=sys.stderr)
            if enc.err:
                print(enc.err, file=sys.stderr)
            return 1
        if len(enc.out) != RECORD_BYTES * 2:
            print(f"FAIL sanity[{i}] encoded length={len(enc.out)}", file=sys.stderr)
            return 1

        ok, dec = decode_should_match(cwd, enc.out, exp)
        if not ok:
            print(f"FAIL sanity[{i}] decode mismatch rc={dec.rc}", file=sys.stderr)
            print(f"payload_hex={payload_hex}", file=sys.stderr)
            print(f"expected={exp}", file=sys.stderr)
            print(f"got={dec.out}", file=sys.stderr)
            if dec.err:
                print(dec.err, file=sys.stderr)
            return 1

    # 2) Max-theoretical correction trials at chosen t (default 73).
    print(f"corruption_trials={args.trials} t={args.t}")
    for i in range(args.trials):
        payload_hex = random_payload_hex(rng)
        exp = expected_decode_hex(payload_hex)

        enc = run_cmd(["./a.out", "enc", payload_hex], cwd)
        if enc.rc != 0 or len(enc.out) != RECORD_BYTES * 2:
            print(f"FAIL corr[{i}] encode rc={enc.rc} len={len(enc.out)}", file=sys.stderr)
            print(f"payload_hex={payload_hex}", file=sys.stderr)
            if enc.err:
                print(enc.err, file=sys.stderr)
            return 1

        mut_hex, muts = mutate_record_hex(enc.out, args.t, rng)
        ok, dec = decode_should_match(cwd, mut_hex, exp)
        if not ok:
            print(f"FAIL corr[{i}] decode mismatch rc={dec.rc}", file=sys.stderr)
            print(f"seed={seed}", file=sys.stderr)
            print(f"payload_hex={payload_hex}", file=sys.stderr)
            print(f"expected={exp}", file=sys.stderr)
            print(f"encoded={enc.out}", file=sys.stderr)
            print(f"mutated={mut_hex}", file=sys.stderr)
            print("mutations(pos:old->new)=" + ",".join(f"{p}:{o:02x}->{n:02x}" for p, o, n in muts), file=sys.stderr)
            print(f"decoder_out={dec.out}", file=sys.stderr)
            if dec.err:
                print(f"decoder_err={dec.err}", file=sys.stderr)
            return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
[/details]

The codec is easy to translate into Rust and trivial to validate through random error injection, so that AI agents can take care of its implementation end-to-end. This is a superior solution to just storing dual CRC-protected copies because it can tolerate damaged writes straddling the two copies (in the original implementation this would cause complete data loss).
