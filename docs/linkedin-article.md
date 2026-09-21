# I benchmarked TypeSafe's JEV against LLMs, BERT and Laya on 1,500 examples. Here's what I found.

Last week TypeSafe AI launched JEV, a "System One" model. It doesn't generate text. You send it your input plus typed questions, and it returns a probability for each option in a single pass. The pitch: LLM-level classification at a fraction of the latency and cost, with no hallucination and no output parsing.

That's a big claim, and most of what I read about it was either the press release or someone's five-example demo. So I built a small benchmark and ran it properly.

## The setup

Six classifiers, three public datasets, 500 held-out examples each, same label descriptions for every zero-shot model.

The datasets, chosen to span difficulty:
- SST-2: movie sentiment, 2 labels
- AG News: news topic, 4 labels
- Banking77: customer intent in a banking app, 77 labels

The contenders:
- JEV (typesafe/jev-1.13) via OpenRouter's Decisions API
- Claude Sonnet 5 and GPT-5-mini via OpenRouter, with JSON-schema constrained output
- Laya, Convai's new open-weights answer to JEV, running locally on my Mac
- DistilBERT fine-tuned per dataset on 10k training examples, also local
- BART-large-MNLI zero-shot as the classic no-training baseline

I measured accuracy, calibration, p50/p95 latency with strictly sequential requests, throughput at fixed concurrency, and cost from the providers' own usage reports. Total API spend: under $10.

## The results

Accuracy and p95 latency:

| Classifier | SST-2 | AG News | Banking77 | p95 latency |
|---|---|---|---|---|
| DistilBERT fine-tuned | 91.0 | 91.0 | 88.0 | 8 to 13 ms |
| JEV | 95.4 | 85.8 | 76.4 | ~670 ms |
| Claude Sonnet 5 | 95.6 | 89.6 | 77.4 | ~2,500 ms |
| GPT-5-mini | 95.0 | 80.2 | 73.6 | ~2,000 ms |
| Laya (local) | 92.0 | 90.6 | 38.2 | 50 to 140 ms |
| BART zero-shot | 89.6 | 76.4 | 42.8 | 70 to 3,000 ms |

Cost per 1,000 classifications on Banking77: JEV $0.08, GPT-5-mini $0.38, Claude Sonnet 5 $6.43. The local models cost nothing per call.

## Seven things I took away

**1. JEV really does deliver frontier-LLM accuracy for a fraction of the price.** Across all three datasets it landed within 0.2 to 4 points of Claude Sonnet 5. On the hardest dataset the gap was one point, inside the noise. It did that at a quarter of the p95 latency and at 50 to 77x lower cost. It beat GPT-5-mini outright on the two harder tasks. If you are picking between JEV and an LLM for a classification step, JEV wins on every axis except the last couple of accuracy points against the most expensive model.

**2. It does not replace a fine-tuned classifier when you have labels.** On Banking77, the cleanest dataset in the set, a fine-tuned DistilBERT beat JEV by 12 points, at 50x lower latency, for zero API cost. The price was 10 minutes of GPU time and 10k labelled examples. If you already have labelled data, a small fine-tuned encoder is still the strongest and cheapest option. JEV's value is when you have no labels, need to ship today, or your label set keeps changing.

**3. JEV's latency is flat, and higher than advertised.** p50 was 380 ms and p95 around 670 ms whether I gave it 2 labels or 77. That consistency is a real strength: cost scales with label count, latency doesn't. But TypeSafe quotes 70 to 100 ms, and through OpenRouter I got 4 to 5x that. If latency matters for you, test the direct API.

**4. Label count is where the field separates.** At 2 labels, all six models landed within 6 points of each other. At 77 labels, three tiers appeared: the fine-tuned model at 88, JEV and the LLMs at 74 to 77, and the small local zero-shot models collapsing to around 40. Holding 77 described options in one pass without degrading is what sets JEV apart from its open-weights lookalike.

**5. Laya is a JEV substitute for small label sets only.** On 2 and 4 labels it matched or beat JEV, ran in 50 to 70 ms locally, and cost nothing. On 77 labels it fell to 38% with a calibration error of 0.51, meaning confidently wrong. For intent routing with under 10 intents, it deserves a serious look. For fine-grained classification, not yet.

**6. Dataset quality shaped the ranking as much as model quality.** AG News has famously inconsistent Business versus Sci/Tech labels. I found "Greenspan: debt, home prices not dangerous" labelled Sci/Tech and "Oracle quarterly net income rises" labelled Business. The three models that scored around 90 there almost certainly learned that noise, either from fine-tuning or from seeing the dataset in pretraining. JEV's 86 is close to the ceiling for a model that follows the label descriptions honestly. When I rewrote the descriptions to match the dataset's convention, JEV gained only 1.5 points because the labels themselves don't follow a rule.

**7. Calibration is the quiet differentiator.** JEV's expected calibration error ranged from 0.03 to 0.13, on par with the fine-tuned model. That means you can threshold on confidence and route uncertain cases to a fallback, which is the pattern that actually matters in production. The LLMs return a label and nothing else.

## Caveats, because they matter

At n=500 the confidence interval on accuracy is roughly plus or minus 2.5 points, so several of these rankings are ties. The fine-tuned model saw 10k labelled examples that nothing else saw. SST-2 and AG News are old enough that the LLMs have almost certainly seen them in pretraining. Banking77 is the table I trust most. The table I'd trust more is one built on my own production data, which is the next thing I'm running.

## The bottom line

If you have labelled data, fine-tune a small encoder. If you don't, or your labels change weekly, JEV gives you frontier-LLM accuracy at cheap-LLM prices with calibrated probabilities on top. And keep an eye on the open-weights models chasing it, because on small label sets they're already there.

The benchmark code is a small Python project with pluggable datasets and classifiers. Happy to share if there's interest.

#MachineLearning #LLM #AIEngineering #TextClassification #JEV
