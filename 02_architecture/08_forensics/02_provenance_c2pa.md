# Provenance / C2PA Architecture

## Purpose

Verify content provenance using C2PA manifests and digital watermarks. Checks whether content has been signed, where it came from, and whether it has been tampered with. First line of defense before running expensive forensic detectors.

---

## Interfaces

### Input

```
ProvenanceInput {
  media_type:      Enum                     # IMAGE | VIDEO | AUDIO
  content:         bytes                    # raw media data
  file_path:       string                   # path to file (optional)
  url:             string                   # URL (optional)
}
```

### Output

```
ProvenanceOutput {
  has_manifest:    bool
  manifest_valid:  bool
  signature_valid: bool
  issuer:          string                   # signing organization
  issued_at:       uint64
  certificate_chain: string[]               # X.509 chain
  claims:          ProvenanceClaim[]
  watermark_detected: bool
  watermark_type:  string                   # "SynthID" | "ContentCredentials" | "unknown"
  tampering_detected: bool
  confidence:      float
  latency_ms:      float
}

ProvenanceClaim {
  claim_type:      string                   # "creation" | "editing" | "ai_generation"
  assertion:       string                   # specific claim
  value:           string
  certificate_id:  string
}
```

### API

```
check_provenance(input: ProvenanceInput) -> ProvenanceOutput
verify_manifest(content: bytes, manifest: bytes) -> bool
extract_watermark(content: bytes) -> WatermarkData
```

---

## Data Contracts

### C2PA v2.4 (April 2026) Formats

| Format | Container | Use Case |
|---|---|---|
| CBOR | Binary | Compact manifest encoding |
| JSON-LD | Text | Interoperability, debugging |
| JUMBF | JPEG | JPEG container |
| BMFF | MP4 | Video container, live streaming |

### C2PA v2.4 Live Video:
- Per-segment manifest boxes for streaming.
- Verifiable Segment Info method for lightweight verification.
- Merkle hash tree for chunk verification.

### SynthID (Google):

| Property | Value |
|---|---|
| Scale | 100B+ images watermarked |
| Audio | 60,000 years of audio |
| Verification | 50M checks in Gemini |
| Robustness | Survives re-encoding, screenshots, ~20% cropping |
| Fragmentation | Google's detector doesn't recognize OpenAI's watermark |

### Configuration

```
ProvenanceConfig {
  check_c2pa:      bool                     # default true
  check_synthid:   bool                     # default true
  check_content_credentials: bool            # default true
  max_latency_ms:  int                      # default 500
  trust_list:      string[]                 # trusted issuers
}
```

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| C2PA manifest extraction | 50–100ms | Parse CBOR/JSON |
| Signature verification | 10–50ms | X.509 chain validation |
| SynthID detection | 50–100ms | Watermark extraction |
| Content Credentials | 30–80ms | Adobe format |
| **Total** | **50–200ms** | |

---

## Dependencies

### Upstream
- `10_infrastructure/01_data_schemas.md` — Media format handling

### Downstream
- `08_forensics/01_deepfake_detection.md` — Skip detectors if provenance verified
- `06_calibration/04_claim_output.md` — Provenance evidence

---

## Reality Check 2026

### C2PA Security (IACR ePrint 2026/804):
- **7 serious problems** found in formal analysis.
- Validators accept manifests signed by known compromised certificates.
- "Exclusion range" allows undetectable alterations.
- v2.4 "does not resolve any of our concerns."
- **At best, assuming valid certificates, C2PA achieves tamper evidence of claims and weak file integrity.**

### Real-World C2PA Preservation (Jun 2026, 520 assets):
- 60% preserved C2PA XMP block on upload.
- 22% natively recognized as 'signed' or 'verified'.
- Only 3 of 9 platforms consistently preserved credentials.
- **C2PA is fragile in practice.** Design for graceful degradation.

### SynthID Adoption:
- OpenAI (ChatGPT, Codex) rolling out SynthID.
- NVIDIA Cosmos carries SynthID since Jan 2026.
- ElevenLabs carries SynthID for audio.
- **Fragmentation:** No cross-provider watermark detection.

### Design Rules:
1. **Check provenance FIRST** — cheapest signal.
2. **Don't trust C2PA alone** — IACR analysis shows vulnerabilities.
3. **Don't trust SynthID alone** — fragmentation across providers.
4. **Use provenance as evidence, not proof** — report in claims with confidence.
