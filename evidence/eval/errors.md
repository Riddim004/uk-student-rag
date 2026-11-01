# Error analysis — test split

Commit: `7b88dec` | corpus: `e12e0d7011f1d5c1` | eval: `7a45aea123b56ee9`

Each row is a query whose retrieval or gate outcome was not a clean hit.
`relevant` is the labelled set; `retrieved` is the top-3 ranked chunk ids.

## bm25

| Error type | Count |
|---|---|
| hit | 10 |
| partial_hit | 4 |
| ranking_error | 4 |
| retrieval_miss | 3 |
| over_refusal | 0 |
| correct_refusal | 0 |
| unanswerable_answered | 4 |
| unanswerable_skipped | 0 |

### partial_hit (4)

- `q20` [paraphrase] What should I do if my eVisa details are wrong?
  - retrieved: `['evisa__report-error-evisa#0', 'evisa__update-ukvi-account#1', 'evisa__view-evisa-get-share-code-prove-immigration-status#0']`
  - relevant: `['evisa__report-error-evisa#0', 'evisa__report-error-evisa#1']`
  - reference: Report an error. UKVI aims to fix most issues within 5 working days, and up to 15 working days for complex cases.
- `q26` [numeric] How long before my course can I apply for a Student visa from inside the UK?
  - retrieved: `['student-visa__overview#0', 'student-visa__overview#2', 'student-visa__overview#1']`
  - relevant: `['student-visa__extend-your-visa#2', 'student-visa__overview#0']`
  - reference: 3 months before the course starts.
- `q28` [numeric] How long does a Graduate visa decision usually take?
  - retrieved: `['graduate-visa__apply#1', 'student-visa__apply-online#0', 'student-visa__overview#1']`
  - relevant: `['graduate-visa__apply#1', 'graduate-visa__overview#1', 'graduate-visa__overview#2']`
  - reference: Usually 8 weeks.
- `q30` [multi_hop] I'm a Chinese national applying for a one-year master's in London. Do I need to show maintenance funds upfront, and how much per month?
  - retrieved: `['student-visa__money#0', 'immigration-rules-student__dependants-of-a-student#5', 'immigration-rules-student__dependants-of-a-student#7']`
  - relevant: `['financial-evidence__nationalities-that-do-not-need-to-provide-financial-evidence#0', 'student-visa__money#0', 'student-visa__money#1', 'student-visa__money#2']`
  - reference: China is on the differential evidence list, so you may not need to provide evidence upfront unless asked; the London maintenance amount is £1,529 per month.

### ranking_error (4)

- `q02` [exact] How much does a Student visa application cost from outside the UK?
  - retrieved: `['ihs__how-much-pay#0', 'graduate-visa__how-much-it-costs#0', 'ihs__how-much-pay#1']`
  - relevant: `['student-visa__overview#2']`
  - reference: £558.
- `q16` [paraphrase] Can my partner join me if I'm studying a one-year taught master's starting in 2026?
  - retrieved: `['graduate-visa__your-partner-and-children#2', 'student-visa__knowledge-of-english#1', 'atas__when-you-need-an-atas-certificate#0']`
  - relevant: `['student-visa__family-members#0']`
  - reference: Generally no. For a postgraduate course starting on or after 1 January 2024, the course must be a PhD/doctorate or research-based higher degree lasting 9 months or longer; a taught master's does not qualify.
- `q22` [numeric] How much extra money do I need for a dependant outside London?
  - retrieved: `['student-visa__money#0', 'financial-evidence__how-much-money-you-need-student-visa#0', 'financial-evidence__how-much-money-you-need-child-student-visa#0']`
  - relevant: `['financial-evidence__how-much-money-you-need-student-visa#0', 'student-visa__family-members#1']`
  - reference: £680 per month for up to 9 months.
- `q50` [chinese] ATAS 认证需要多长时间？
  - retrieved: `['atas__outcome-of-your-application#0', 'atas__contacting-the-atas-team#0', 'atas__students-further-information#1']`
  - relevant: `['atas__the-application-process#0']`
  - reference: 至少 30 个工作日（6 周），复杂申请可能更久。

### retrieval_miss (3)

- `q44` [chinese] 伦敦地区每月生活费需要准备多少钱？
  - retrieved: `[]`
  - relevant: `['financial-evidence__how-much-money-you-need-student-visa#0', 'student-visa__money#0']`
  - reference: 每月 £1,529，最多 9 个月。
- `q46` [chinese] 博士毕业后毕业生签证可以停留多久？
  - retrieved: `[]`
  - relevant: `['graduate-visa__overview#0']`
  - reference: 博士或其他博士学位可获得 3 年毕业生签证。
- `q48` [chinese] 我可以带配偶和孩子一起去英国吗？
  - retrieved: `[]`
  - relevant: `['student-visa__family-members#0']`
  - reference: 取决于课程类型和资助情况；通常需为政府资助的 6 个月以上课程，或 9 个月以上的全日制研究生课程，且 2024 年 1 月 1 日后开课的课程须为博士或研究型高等学位。

### unanswerable_answered (4)

- `q39` [unanswerable] What is the minimum salary for a Skilled Worker visa?
  - retrieved: `['immigration-rules-student__work-conditions-for-a-student#1', 'visa-processing-times__work-visas#0', 'atas__contracted-researchers-further-information#0']`
  - relevant: `[]`
  - reference: The corpus mentions the Skilled Worker route but does not state its salary threshold. The correct behaviour is refusal.
- `q40` [unanswerable] How do I apply for British citizenship after my Student visa?
  - retrieved: `['student-visa__extend-your-visa#3', 'student-visa__switch-to-this-visa#1', 'graduate-visa__apply#0']`
  - relevant: `[]`
  - reference: The corpus does not cover British citizenship applications. The correct behaviour is refusal.
- `q41` [unanswerable] What tuition fees does the University of York charge for international students?
  - retrieved: `['immigration-rules-student__dependants-of-a-student#0', 'immigration-rules-student__validity-requirements-for-a-student#0', 'immigration-rules-student__eligibility-requirements-for-a-student#8']`
  - relevant: `[]`
  - reference: The corpus is gov.uk visa guidance, not university fee data. The correct behaviour is refusal.
- `q42` [unanswerable] Do I need private health insurance in addition to the Immigration Health Surcharge?
  - retrieved: `['ihs__who-needs-pay#0', 'graduate-visa__overview#2', 'ihs__refunds#2']`
  - relevant: `[]`
  - reference: The corpus does not cover private health insurance. The correct behaviour is refusal.

## dense

| Error type | Count |
|---|---|
| hit | 13 |
| partial_hit | 2 |
| ranking_error | 5 |
| retrieval_miss | 1 |
| over_refusal | 0 |
| correct_refusal | 0 |
| unanswerable_answered | 4 |
| unanswerable_skipped | 0 |

### partial_hit (2)

- `q24` [numeric] How much is the Graduate visa IHS for 3 years?
  - retrieved: `['graduate-visa__how-much-it-costs#0', 'graduate-visa__overview#0', 'ihs__who-needs-pay#2']`
  - relevant: `['graduate-visa__how-much-it-costs#0', 'ihs__how-much-pay#0']`
  - reference: £3,105.
- `q30` [multi_hop] I'm a Chinese national applying for a one-year master's in London. Do I need to show maintenance funds upfront, and how much per month?
  - retrieved: `['student-visa__money#0', 'immigration-rules-student__dependants-of-a-student#5', 'student-visa__money#1']`
  - relevant: `['financial-evidence__nationalities-that-do-not-need-to-provide-financial-evidence#0', 'student-visa__money#0', 'student-visa__money#1', 'student-visa__money#2']`
  - reference: China is on the differential evidence list, so you may not need to provide evidence upfront unless asked; the London maintenance amount is £1,529 per month.

### ranking_error (5)

- `q02` [exact] How much does a Student visa application cost from outside the UK?
  - retrieved: `['student-visa__overview#0', 'student-visa__overview#2', 'student-visa__extend-your-visa#4']`
  - relevant: `['student-visa__overview#2']`
  - reference: £558.
- `q10` [exact] How much is the Immigration Health Surcharge for a student per year?
  - retrieved: `['graduate-visa__how-much-it-costs#0', 'ihs__how-much-pay#0', 'ihs__refunds#2']`
  - relevant: `['ihs__how-much-pay#0']`
  - reference: £776 per year for students.
- `q22` [numeric] How much extra money do I need for a dependant outside London?
  - retrieved: `['student-visa__money#0', 'financial-evidence__how-much-money-you-need-student-visa#0', 'financial-evidence__how-much-money-you-need-child-student-visa#0']`
  - relevant: `['financial-evidence__how-much-money-you-need-student-visa#0', 'student-visa__family-members#1']`
  - reference: £680 per month for up to 9 months.
- `q28` [numeric] How long does a Graduate visa decision usually take?
  - retrieved: `['graduate-visa__overview#0', 'graduate-visa__apply#1', 'graduate-visa__overview#1']`
  - relevant: `['graduate-visa__apply#1', 'graduate-visa__overview#1', 'graduate-visa__overview#2']`
  - reference: Usually 8 weeks.
- `q48` [chinese] 我可以带配偶和孩子一起去英国吗？
  - retrieved: `['graduate-visa__your-partner-and-children#4', 'student-visa__family-members#4', 'student-visa__family-members#5']`
  - relevant: `['student-visa__family-members#0']`
  - reference: 取决于课程类型和资助情况；通常需为政府资助的 6 个月以上课程，或 9 个月以上的全日制研究生课程，且 2024 年 1 月 1 日后开课的课程须为博士或研究型高等学位。

### retrieval_miss (1)

- `q14` [paraphrase] Can I be self-employed on a Graduate visa?
  - retrieved: `['graduate-visa__overview#3', 'graduate-visa__overview#0', 'graduate-visa__apply#0']`
  - relevant: `['graduate-visa__overview#2']`
  - reference: Yes, self-employment is allowed on a Graduate visa.

### unanswerable_answered (4)

- `q39` [unanswerable] What is the minimum salary for a Skilled Worker visa?
  - retrieved: `['ihs__who-needs-pay#1', 'visa-processing-times__other-visas-and-permits#0', 'visa-processing-times__work-visas#1']`
  - relevant: `[]`
  - reference: The corpus mentions the Skilled Worker route but does not state its salary threshold. The correct behaviour is refusal.
- `q40` [unanswerable] How do I apply for British citizenship after my Student visa?
  - retrieved: `['student-visa__extend-your-visa#3', 'student-visa__extend-your-visa#0', 'student-visa__switch-to-this-visa#0']`
  - relevant: `[]`
  - reference: The corpus does not cover British citizenship applications. The correct behaviour is refusal.
- `q41` [unanswerable] What tuition fees does the University of York charge for international students?
  - retrieved: `['immigration-rules-student__eligibility-requirements-for-a-student#8', 'ihs__how-much-pay#0', 'student-visa__money#2']`
  - relevant: `[]`
  - reference: The corpus is gov.uk visa guidance, not university fee data. The correct behaviour is refusal.
- `q42` [unanswerable] Do I need private health insurance in addition to the Immigration Health Surcharge?
  - retrieved: `['ihs__overview#0', 'ihs__who-needs-pay#0', 'ihs__who-needs-pay#2']`
  - relevant: `[]`
  - reference: The corpus does not cover private health insurance. The correct behaviour is refusal.

## hybrid

| Error type | Count |
|---|---|
| hit | 14 |
| partial_hit | 2 |
| ranking_error | 5 |
| retrieval_miss | 0 |
| over_refusal | 0 |
| correct_refusal | 0 |
| unanswerable_answered | 4 |
| unanswerable_skipped | 0 |

### partial_hit (2)

- `q28` [numeric] How long does a Graduate visa decision usually take?
  - retrieved: `['graduate-visa__apply#1', 'graduate-visa__overview#1', 'student-visa__overview#1']`
  - relevant: `['graduate-visa__apply#1', 'graduate-visa__overview#1', 'graduate-visa__overview#2']`
  - reference: Usually 8 weeks.
- `q30` [multi_hop] I'm a Chinese national applying for a one-year master's in London. Do I need to show maintenance funds upfront, and how much per month?
  - retrieved: `['student-visa__money#0', 'immigration-rules-student__dependants-of-a-student#5', 'immigration-rules-student__dependants-of-a-student#7']`
  - relevant: `['financial-evidence__nationalities-that-do-not-need-to-provide-financial-evidence#0', 'student-visa__money#0', 'student-visa__money#1', 'student-visa__money#2']`
  - reference: China is on the differential evidence list, so you may not need to provide evidence upfront unless asked; the London maintenance amount is £1,529 per month.

### ranking_error (5)

- `q02` [exact] How much does a Student visa application cost from outside the UK?
  - retrieved: `['ihs__how-much-pay#0', 'student-visa__overview#2', 'financial-evidence__how-much-money-you-need-student-visa#0']`
  - relevant: `['student-visa__overview#2']`
  - reference: £558.
- `q14` [paraphrase] Can I be self-employed on a Graduate visa?
  - retrieved: `['graduate-visa__overview#0', 'graduate-visa__overview#3', 'graduate-visa__overview#1']`
  - relevant: `['graduate-visa__overview#2']`
  - reference: Yes, self-employment is allowed on a Graduate visa.
- `q22` [numeric] How much extra money do I need for a dependant outside London?
  - retrieved: `['student-visa__money#0', 'financial-evidence__how-much-money-you-need-student-visa#0', 'financial-evidence__how-much-money-you-need-child-student-visa#0']`
  - relevant: `['financial-evidence__how-much-money-you-need-student-visa#0', 'student-visa__family-members#1']`
  - reference: £680 per month for up to 9 months.
- `q48` [chinese] 我可以带配偶和孩子一起去英国吗？
  - retrieved: `['graduate-visa__your-partner-and-children#4', 'student-visa__family-members#4', 'student-visa__family-members#5']`
  - relevant: `['student-visa__family-members#0']`
  - reference: 取决于课程类型和资助情况；通常需为政府资助的 6 个月以上课程，或 9 个月以上的全日制研究生课程，且 2024 年 1 月 1 日后开课的课程须为博士或研究型高等学位。
- `q50` [chinese] ATAS 认证需要多长时间？
  - retrieved: `['atas__contacting-the-atas-team#0', 'atas__outcome-of-your-application#0', 'atas__the-application-process#0']`
  - relevant: `['atas__the-application-process#0']`
  - reference: 至少 30 个工作日（6 周），复杂申请可能更久。

### unanswerable_answered (4)

- `q39` [unanswerable] What is the minimum salary for a Skilled Worker visa?
  - retrieved: `['visa-processing-times__work-visas#0', 'graduate-visa__overview#1', 'atas__when-you-need-an-atas-certificate#0']`
  - relevant: `[]`
  - reference: The corpus mentions the Skilled Worker route but does not state its salary threshold. The correct behaviour is refusal.
- `q40` [unanswerable] How do I apply for British citizenship after my Student visa?
  - retrieved: `['student-visa__extend-your-visa#3', 'student-visa__switch-to-this-visa#1', 'student-visa__apply-online#0']`
  - relevant: `[]`
  - reference: The corpus does not cover British citizenship applications. The correct behaviour is refusal.
- `q41` [unanswerable] What tuition fees does the University of York charge for international students?
  - retrieved: `['immigration-rules-student__eligibility-requirements-for-a-student#8', 'ihs__how-much-pay#0', 'student-visa__overview#2']`
  - relevant: `[]`
  - reference: The corpus is gov.uk visa guidance, not university fee data. The correct behaviour is refusal.
- `q42` [unanswerable] Do I need private health insurance in addition to the Immigration Health Surcharge?
  - retrieved: `['ihs__who-needs-pay#0', 'ihs__overview#0', 'graduate-visa__overview#2']`
  - relevant: `[]`
  - reference: The corpus does not cover private health insurance. The correct behaviour is refusal.

## hybrid+rerank

| Error type | Count |
|---|---|
| hit | 13 |
| partial_hit | 0 |
| ranking_error | 8 |
| retrieval_miss | 0 |
| over_refusal | 0 |
| correct_refusal | 2 |
| unanswerable_answered | 2 |
| unanswerable_skipped | 0 |

### ranking_error (8)

- `q12` [paraphrase] How much cash do I need in my bank account to cover living costs in London?
  - retrieved: `['immigration-rules-student__dependants-of-a-student#7', 'immigration-rules-student__dependants-of-a-student#5', 'financial-evidence__how-much-money-you-need-child-student-visa#0']`
  - relevant: `['financial-evidence__how-much-money-you-need-student-visa#0', 'student-visa__money#0']`
  - reference: £1,529 per month for up to 9 months for courses in London.
- `q14` [paraphrase] Can I be self-employed on a Graduate visa?
  - retrieved: `['immigration-rules-student__work-conditions-for-a-student#1', 'graduate-visa__overview#2', 'student-visa__overview#3']`
  - relevant: `['graduate-visa__overview#2']`
  - reference: Yes, self-employment is allowed on a Graduate visa.
- `q26` [numeric] How long before my course can I apply for a Student visa from inside the UK?
  - retrieved: `['student-visa__overview#2', 'student-visa__overview#0', 'student-visa__overview#1']`
  - relevant: `['student-visa__extend-your-visa#2', 'student-visa__overview#0']`
  - reference: 3 months before the course starts.
- `q28` [numeric] How long does a Graduate visa decision usually take?
  - retrieved: `['graduate-visa__your-partner-and-children#3', 'graduate-visa__apply#1', 'student-visa__apply-online#1']`
  - relevant: `['graduate-visa__apply#1', 'graduate-visa__overview#1', 'graduate-visa__overview#2']`
  - reference: Usually 8 weeks.
- `q30` [multi_hop] I'm a Chinese national applying for a one-year master's in London. Do I need to show maintenance funds upfront, and how much per month?
  - retrieved: `['immigration-rules-student__dependants-of-a-student#7', 'immigration-rules-student__dependants-of-a-student#5', 'immigration-rules-student__eligibility-requirements-for-a-student#8']`
  - relevant: `['financial-evidence__nationalities-that-do-not-need-to-provide-financial-evidence#0', 'student-visa__money#0', 'student-visa__money#1', 'student-visa__money#2']`
  - reference: China is on the differential evidence list, so you may not need to provide evidence upfront unless asked; the London maintenance amount is £1,529 per month.
- `q44` [chinese] 伦敦地区每月生活费需要准备多少钱？
  - retrieved: `['immigration-rules-student__dependants-of-a-student#7', 'immigration-rules-student__dependants-of-a-student#5', 'immigration-rules-student__eligibility-requirements-for-a-student#8']`
  - relevant: `['financial-evidence__how-much-money-you-need-student-visa#0', 'student-visa__money#0']`
  - reference: 每月 £1,529，最多 9 个月。
- `q46` [chinese] 博士毕业后毕业生签证可以停留多久？
  - retrieved: `['graduate-visa__overview#1', 'graduate-visa__overview#0', 'student-visa__overview#1']`
  - relevant: `['graduate-visa__overview#0']`
  - reference: 博士或其他博士学位可获得 3 年毕业生签证。
- `q50` [chinese] ATAS 认证需要多长时间？
  - retrieved: `['atas__atas-immigration-rules#0', 'atas__the-application-process#0', 'atas__outcome-of-your-application#0']`
  - relevant: `['atas__the-application-process#0']`
  - reference: 至少 30 个工作日（6 周），复杂申请可能更久。

### unanswerable_answered (2)

- `q40` [unanswerable] How do I apply for British citizenship after my Student visa?
  - retrieved: `['student-visa__family-members#5', 'student-visa__extend-your-visa#4', 'student-visa__switch-to-this-visa#2']`
  - relevant: `[]`
  - reference: The corpus does not cover British citizenship applications. The correct behaviour is refusal.
- `q42` [unanswerable] Do I need private health insurance in addition to the Immigration Health Surcharge?
  - retrieved: `['ihs__who-needs-pay#0', 'ihs__overview#0', 'ihs__who-needs-pay#1']`
  - relevant: `[]`
  - reference: The corpus does not cover private health insurance. The correct behaviour is refusal.

## hybrid+rerank-gate

| Error type | Count |
|---|---|
| hit | 14 |
| partial_hit | 2 |
| ranking_error | 5 |
| retrieval_miss | 0 |
| over_refusal | 0 |
| correct_refusal | 2 |
| unanswerable_answered | 2 |
| unanswerable_skipped | 0 |

### partial_hit (2)

- `q28` [numeric] How long does a Graduate visa decision usually take?
  - retrieved: `['graduate-visa__apply#1', 'graduate-visa__overview#1', 'student-visa__overview#1']`
  - relevant: `['graduate-visa__apply#1', 'graduate-visa__overview#1', 'graduate-visa__overview#2']`
  - reference: Usually 8 weeks.
- `q30` [multi_hop] I'm a Chinese national applying for a one-year master's in London. Do I need to show maintenance funds upfront, and how much per month?
  - retrieved: `['student-visa__money#0', 'immigration-rules-student__dependants-of-a-student#5', 'immigration-rules-student__dependants-of-a-student#7']`
  - relevant: `['financial-evidence__nationalities-that-do-not-need-to-provide-financial-evidence#0', 'student-visa__money#0', 'student-visa__money#1', 'student-visa__money#2']`
  - reference: China is on the differential evidence list, so you may not need to provide evidence upfront unless asked; the London maintenance amount is £1,529 per month.

### ranking_error (5)

- `q02` [exact] How much does a Student visa application cost from outside the UK?
  - retrieved: `['ihs__how-much-pay#0', 'student-visa__overview#2', 'financial-evidence__how-much-money-you-need-student-visa#0']`
  - relevant: `['student-visa__overview#2']`
  - reference: £558.
- `q14` [paraphrase] Can I be self-employed on a Graduate visa?
  - retrieved: `['graduate-visa__overview#0', 'graduate-visa__overview#3', 'graduate-visa__overview#1']`
  - relevant: `['graduate-visa__overview#2']`
  - reference: Yes, self-employment is allowed on a Graduate visa.
- `q22` [numeric] How much extra money do I need for a dependant outside London?
  - retrieved: `['student-visa__money#0', 'financial-evidence__how-much-money-you-need-student-visa#0', 'financial-evidence__how-much-money-you-need-child-student-visa#0']`
  - relevant: `['financial-evidence__how-much-money-you-need-student-visa#0', 'student-visa__family-members#1']`
  - reference: £680 per month for up to 9 months.
- `q48` [chinese] 我可以带配偶和孩子一起去英国吗？
  - retrieved: `['graduate-visa__your-partner-and-children#4', 'student-visa__family-members#4', 'student-visa__family-members#5']`
  - relevant: `['student-visa__family-members#0']`
  - reference: 取决于课程类型和资助情况；通常需为政府资助的 6 个月以上课程，或 9 个月以上的全日制研究生课程，且 2024 年 1 月 1 日后开课的课程须为博士或研究型高等学位。
- `q50` [chinese] ATAS 认证需要多长时间？
  - retrieved: `['atas__contacting-the-atas-team#0', 'atas__outcome-of-your-application#0', 'atas__the-application-process#0']`
  - relevant: `['atas__the-application-process#0']`
  - reference: 至少 30 个工作日（6 周），复杂申请可能更久。

### unanswerable_answered (2)

- `q40` [unanswerable] How do I apply for British citizenship after my Student visa?
  - retrieved: `['student-visa__extend-your-visa#3', 'student-visa__switch-to-this-visa#1', 'student-visa__apply-online#0']`
  - relevant: `[]`
  - reference: The corpus does not cover British citizenship applications. The correct behaviour is refusal.
- `q42` [unanswerable] Do I need private health insurance in addition to the Immigration Health Surcharge?
  - retrieved: `['ihs__who-needs-pay#0', 'ihs__overview#0', 'graduate-visa__overview#2']`
  - relevant: `[]`
  - reference: The corpus does not cover private health insurance. The correct behaviour is refusal.
