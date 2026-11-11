# Extraction Accuracy

Gold labels: `eval/extraction_gold.json` (checked by hand against the documents). A field is correct when the value contains the expected facts, or is null when the documents do not state it. Ambiguous fields and the generated summary are not scored.

| Mode | Bid1 | Bid2 | Total |
|---|---|---|---|
| rules | 17/18 | 18/18 | 35/36 (97%) |

## Per field

| Bid | Field | rules |
|---|---|---|
| Bid1 | Bid Number | ok |
| Bid1 | Title | ok |
| Bid1 | Due Date | ok |
| Bid1 | Bid Submission Type | ok |
| Bid1 | Term of Bid | ok |
| Bid1 | Pre Bid Meeting | ok |
| Bid1 | Installation | ok |
| Bid1 | Bid Bond Requirement | ok |
| Bid1 | Delivery Date | **wrong** |
| Bid1 | Payment Terms | ok |
| Bid1 | Any Additional Documentation Required | ok |
| Bid1 | MFG for Registration | - |
| Bid1 | Contract or Cooperative to use | ok |
| Bid1 | Model_no | ok |
| Bid1 | Part_no | ok |
| Bid1 | Product | ok |
| Bid1 | contact_info | ok |
| Bid1 | company_name | ok |
| Bid1 | Product Specification | ok |
| Bid1 | Bid Summary | - |
| Bid2 | Bid Number | ok |
| Bid2 | Title | ok |
| Bid2 | Due Date | ok |
| Bid2 | Bid Submission Type | ok |
| Bid2 | Term of Bid | ok |
| Bid2 | Pre Bid Meeting | ok |
| Bid2 | Installation | - |
| Bid2 | Bid Bond Requirement | ok |
| Bid2 | Delivery Date | ok |
| Bid2 | Payment Terms | ok |
| Bid2 | Any Additional Documentation Required | ok |
| Bid2 | MFG for Registration | ok |
| Bid2 | Contract or Cooperative to use | ok |
| Bid2 | Model_no | ok |
| Bid2 | Part_no | ok |
| Bid2 | Product | ok |
| Bid2 | contact_info | ok |
| Bid2 | company_name | ok |
| Bid2 | Product Specification | ok |
| Bid2 | Bid Summary | - |
