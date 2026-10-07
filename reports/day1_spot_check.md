# Day 1 Spot-Check Report (v2) — Egyptian Civil Code Ingestion

Source: `data/raw/civil_code.pdf` (uploaded by user, `llm_final_pro.pdf`)
Output checked: `data/processed/civil_code.json`

This is a follow-up pass after the first spot check. It focuses on: (1)
resolving the 7 previously-unexplained individual gaps, (2) fixing the
false duplicate at article 1148, and (3) stripping heading text out of
`text_ar`/`text_en`.

## 1. Individual gap investigation (277, 452, 714, 746, 898, 1022, 1090)

All 7 were searched for directly in the raw extracted PDF text.

| # | Found in PDF? | Root cause | Resolution |
|---|---|---|---|
| 277 | Yes | Heading heuristic rejected it: the true heading line ("Article 277 If the option belongs to the debtor and not مادة ٢٧٧") has a trailing English sentence fragment between the English and Arabic markers, which looked like a mid-sentence citation. | Fixed — heuristic now accepts a heading when the English marker opens the line and both markers agree on the number, regardless of what's between them. Article now correctly parsed. |
| 452 | Yes | Heading line was "rticle 452 مادة ٤٥٢" — the PDF extraction dropped the leading "A" of "Article" for this one occurrence, so the English-marker regex didn't match. | Fixed — English marker regex now tolerates a missing leading "A" (`A?rticle`). |
| 714 | Yes | Same root cause as 277 (trailing English sentence before Arabic marker). | Fixed, same heuristic change. |
| 746 | Yes | Same root cause as 277. | Fixed, same heuristic change. |
| 898 | Yes | Same root cause as 277. | Fixed, same heuristic change. |
| 1090 | Yes | Same root cause as 277. | Fixed, same heuristic change. |
| 1022 | Yes (English only) | The heading line is bare "Article1022" with **no Arabic marker pairing at all** on that line or nearby — a genuine extraction anomaly in the source PDF (compare articles 1021 and 1023, both of which have the normal "Article N مادة N" bilingual pairing). | Article record now created from the English marker alone (new `_find_english_only_heading` fallback, only triggered when the English marker is the entire line). `text_en` is populated; **`text_ar` is correctly left empty** rather than fabricated, and this is flagged by validation (`article_issues: {"1022": ["text_ar is empty"]}"`). **This is the one case that cannot be conclusively resolved from the text alone — marked as an open, unresolved extraction gap, not guessed at.** |

Re-running validation after these fixes: **all 7 numbers are now resolved**
(6 fully, 1 partially with the empty-`text_ar` flag standing in place of a
silent gap). Zero silent/undocumented gaps remain among these seven.

## 2. Remaining number gaps (56 total, all accounted for)

After the fixes above, the *only* remaining gaps are the two ranges the
PDF explicitly marks as repealed:
- **54–80** (27 articles): inline notice `Articles 54-80 ... repealed`.
- **389–417** (29 articles): inline notice
  `Articles 389-417 repealed المواد من ٣٨٩ إلى ٤١٧ ملغاة`.

27 + 29 = 56, matching the validator's gap count exactly. No unexplained
individual gaps remain.

## 3. False duplicate at article 1148 — fixed

Root cause: a short, English-free wrapped line, `.المادة ١١٤٨` (a
line-wrap remainder of a longer sentence citing article 1148, with only a
trailing period as "extra" content), previously slipped past the
Arabic-only heading check because punctuation was stripped before the
emptiness check.

Fix: the Arabic-only branch of the heading heuristic now requires the
match to be the **entire line** (whitespace-trimmed only, no punctuation
stripped) — a bare period now correctly disqualifies it.

Verified: `data/processed/civil_code.json` now has **exactly one**
occurrence of article 1148, at page 170, with intact real content ("Sums
due to contractors and architects who have been ... / المبالغ المستحقة
للمقاولين ..."). Duplicate numbers are now only `[1, 2]` — the legitimate
Promulgation Law / Civil Code numbering restart (documented in the first
spot-check report) — down from `[1, 2, 1148]`.

Regression test added: `test_wrapped_arabic_only_citation_with_trailing_punctuation_rejected`.

### A second-order bug this surfaced and also fixed
Loosening the heading rule for case 277/452/etc. initially reintroduced
5 NEW false duplicates (298, 864, 982, 1033, 1050) — real body sentences
that cite another article by number in both English and Arabic on the
same line (e.g. "with Article 298 claim from the co-debtor who has been
.للمادة ٢٩٨"). Fixed by requiring the English marker to additionally
*open* the line (nothing before it) — true headings always start the
line with "Article N"; these citation sentences always have real text
before it. Regression test:
`test_midsentence_bilingual_cross_reference_is_not_a_new_article`.

## 4. Heading text no longer leaks into text_ar/text_en

The parser now strips the marker text itself ("Article 105 مادة ١٠٥")
out of the heading line before adding any remainder to the article body.
`article_number` already carries the number separately, so nothing is
lost. Any genuine body-text fragment sharing the heading line (real in
this PDF, e.g. article 277's "If the option belongs to the debtor and
not") is preserved verbatim, not discarded.

Verified across a random sample of 15 articles: no heading-marker leakage
found (one line contained the word "المادة" as ordinary body prose --
"the preceding article" -- which is legitimate content, not a leak).
Spot-checked directly: articles 1, 2, 105, 244, 800, 1148, 1149 all now
have `text_ar` starting with real body content, not "Article N مادة N".

Regression tests added:
`test_heading_marker_text_not_included_in_body`,
`test_bilingual_heading_with_trailing_english_sentence_before_arabic_marker`,
`test_english_heading_missing_leading_letter_is_recognized`,
`test_english_only_heading_with_no_arabic_marker_at_all`.

## 5. Re-run results

- **1095 articles** parsed (up from 1089; +6 recovered gap articles).
- Validation: **1 article-level issue** (1022's empty `text_ar`, documented
  above -- not silently dropped), **2 duplicate numbers** (both the
  legitimate front-matter restart), **56 gaps** (both fully explained by
  documented repeal ranges).
- 29 articles re-spot-checked (the original 22 + the 7 fixed gap cases),
  all present with correct numbers, legible Arabic/English text, and no
  heading leakage. Article 1022 confirmed present with English text only
  (Arabic genuinely absent from source).

## Known residual limitations (unchanged from before unless noted)
- Article 1022's `text_ar` is empty -- a genuine, unresolved source
  extraction gap. Flagged by validation, not fabricated.
- Bilingual `text_ar`/`text_en` split is still a per-line Latin-ratio
  heuristic, not a true column-based split.
- `topic` field is still always `None`.
