# SBOL test fixtures

Real-world SBOL files vendored from the SynBioDex **SBOLTestSuite**, the
community conformance corpus for the SBOL standard.

- Source: https://github.com/SynBioDex/SBOLTestSuite
- Pinned commit: see `COMMIT_SHA.txt` (`0044284331b2f915a6e4b9d50e1cbf3ea2f62dcd`)

## Files

| File | Format | Notes |
|------|--------|-------|
| `sbol2/pICH44179.xml` | SBOL2 RDF/XML | A plasmid with one ~2307 bp sequence. |
| `sbol3/BBa_F2620_PoPSReceiver.ttl` | SBOL3 Turtle | iGEM device; 10 sequences. |
| `sbol3/toggle_switch.ttl` | SBOL3 Turtle | Abstract design: components, sub-components, and interactions but **no inline sequences**. |
| `sbol3/toggle_switch.nt` | SBOL3 N-Triples | Same design, different serialization. |

## License

⚠️ The SBOLTestSuite repository does not carry an explicit license file. These
files are SynBioDex's public SBOL conformance examples, included here solely as
test fixtures. Confirm redistribution terms before publishing this repository.
