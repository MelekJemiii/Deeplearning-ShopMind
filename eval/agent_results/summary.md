# RAG Agent — end-to-end evaluation

_31/31 queries answered by the workflow. Router bypassed (empty needs): measures the RAG Agent alone._

**Models that answered:** fallback: groq openai/gpt-oss-120b (31)

## Answer quality (all answered queries)

| Metric | Value |
|---|---|
| In-scope: at least one fact returned | 100% (n=28) |
| In-scope: agent retrieved a relevant document | 93% |
| In-scope: agent **cited** a relevant document | 93% |
| In-scope: wrongly abstained (not_found) | 0% |
| Out-of-scope: correctly abstained | 100% (n=3) |

## By query type (in-scope)

| Type | n | Answered | Cited relevant doc |
|---|---|---|---|
| compatibility | 3 | 100% | 100% |
| cross_lingual | 3 | 100% | 100% |
| definition | 7 | 100% | 71% |
| spec_lookup | 6 | 100% | 100% |
| use_case | 9 | 100% | 100% |

## Guardrails (deterministic validation node)

- Facts removed for citing a chunk never retrieved: **0**
- Facts removed for a number absent from the cited chunk: **0**
- Queries where a removal happened: none

## Search behavior

- Mean tool calls per query: **1.13**
- One search per answered question: 96%

| Query type | source_type chosen by the agent (counts) |
|---|---|
| compatibility | any: 1, manufacturer_pdf: 1, team_written: 2 |
| cross_lingual | manufacturer_pdf: 3 |
| definition | any: 5, team_written: 3, wikipedia: 2 |
| spec_lookup | manufacturer_pdf: 6 |
| use_case | team_written: 12 |

**Latency (end-to-end, per query):** mean 7.1 s, max 22.8 s

## Per query

| id | type | answered | cited relevant doc | tool calls | sources | model | latency (s) |
|---|---|---|---|---|---|---|---|
| q01 | use_case | yes | yes | 1 | team_written | fallback: groq openai/gpt-oss-120b | 6.1 |
| q02 | use_case | yes | yes | 1 | team_written | fallback: groq openai/gpt-oss-120b | 11.4 |
| q03 | use_case | yes | yes | 1 | team_written | fallback: groq openai/gpt-oss-120b | 4.3 |
| q04 | use_case | yes | yes | 2 | team_written, team_written | fallback: groq openai/gpt-oss-120b | 22.8 |
| q05 | use_case | yes | yes | 3 | team_written, team_written, team_written | fallback: groq openai/gpt-oss-120b | 17.5 |
| q06 | use_case | yes | yes | 1 | team_written | fallback: groq openai/gpt-oss-120b | 3.6 |
| q07 | use_case | yes | yes | 1 | team_written | fallback: groq openai/gpt-oss-120b | 5.4 |
| q08 | use_case | yes | yes | 1 | team_written | fallback: groq openai/gpt-oss-120b | 6.7 |
| q09 | use_case | yes | yes | 1 | team_written | fallback: groq openai/gpt-oss-120b | 5.7 |
| q10 | compatibility | yes | yes | 2 | team_written, any | fallback: groq openai/gpt-oss-120b | 19.8 |
| q11 | compatibility | yes | yes | 1 | team_written | fallback: groq openai/gpt-oss-120b | 5.1 |
| q12 | compatibility | yes | yes | 1 | manufacturer_pdf | fallback: groq openai/gpt-oss-120b | 4.7 |
| q13 | definition | yes | yes | 1 | any | fallback: groq openai/gpt-oss-120b | 10.1 |
| q14 | definition | yes | yes | 1 | wikipedia | fallback: groq openai/gpt-oss-120b | 6.0 |
| q15 | definition | yes | no | 1 | any | fallback: groq openai/gpt-oss-120b | 5.3 |
| q16 | definition | yes | yes | 2 | wikipedia, any | fallback: groq openai/gpt-oss-120b | 7.0 |
| q17 | definition | yes | yes | 2 | team_written, any | fallback: groq openai/gpt-oss-120b | 7.8 |
| q18 | definition | yes | no | 2 | any, team_written | fallback: groq openai/gpt-oss-120b | 16.7 |
| q19 | definition | yes | yes | 1 | team_written | fallback: groq openai/gpt-oss-120b | 8.8 |
| q20 | spec_lookup | yes | yes | 1 | manufacturer_pdf | fallback: groq openai/gpt-oss-120b | 4.9 |
| q21 | spec_lookup | yes | yes | 1 | manufacturer_pdf | fallback: groq openai/gpt-oss-120b | 3.5 |
| q22 | spec_lookup | yes | yes | 1 | manufacturer_pdf | fallback: groq openai/gpt-oss-120b | 3.7 |
| q23 | spec_lookup | yes | yes | 1 | manufacturer_pdf | fallback: groq openai/gpt-oss-120b | 4.8 |
| q24 | spec_lookup | yes | yes | 1 | manufacturer_pdf | fallback: groq openai/gpt-oss-120b | 3.8 |
| q25 | spec_lookup | yes | yes | 1 | manufacturer_pdf | fallback: groq openai/gpt-oss-120b | 4.7 |
| q26 | cross_lingual | yes | yes | 1 | manufacturer_pdf | fallback: groq openai/gpt-oss-120b | 3.6 |
| q27 | cross_lingual | yes | yes | 1 | manufacturer_pdf | fallback: groq openai/gpt-oss-120b | 5.8 |
| q28 | cross_lingual | yes | yes | 1 | manufacturer_pdf | fallback: groq openai/gpt-oss-120b | 3.8 |
| q29 | out_of_scope | no | no | 0 | - | fallback: groq openai/gpt-oss-120b | 2.2 |
| q30 | out_of_scope | no | no | 0 | - | fallback: groq openai/gpt-oss-120b | 2.6 |
| q31 | out_of_scope | no | no | 0 | - | fallback: groq openai/gpt-oss-120b | 1.6 |
