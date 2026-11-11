# Assumptions

- The addendum PDF is a text-based PDF (not scanned). OCR is not implemented.
- The 39 numbered items are the only Q&A structure in the document. The parser requires sequential numbering.
- "Answer:" marks the start of each District response.
- Dates in the addendum (2024) describe the original bid. The current status of the bid was not checked.
- Gold labels in `eval/questions.json` reflect the author's reading of each answer. A second reviewer should check them.
- Confidence values are heuristics for ranking evidence, not probabilities.
