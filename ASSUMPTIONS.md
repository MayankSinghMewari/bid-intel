# Assumptions

- A bid folder holds the documents of exactly one bid. Document types are inferred from file names
  (`addendum N`, `affidavit`/`form`/`certif`, `spec`, `.html` bid page, otherwise RFP).
- PDFs contain extractable text. Scanned pages are detected (no text) and logged, not OCR'd.
- An addendum with a higher number supersedes earlier documents for the fields it changes.
- BidNet-style bid pages present "label / value" pairs between "Basic Information" and "Disclaimer";
  other portals still work through the PDF text, without the structured facts.
- Dates are reported as written in the documents (time zones are not converted).
- The bid documents are public procurement records, so traces are not masked.
- Gold labels in `eval/` were written and checked against the documents by the author; a second
  reviewer should check them. Ambiguous fields are marked `skip` with the reason.
- Confidence values are heuristics for ranking and review, not calibrated probabilities.
