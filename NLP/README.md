# Sexism Classifier

## Objective
Classify input sentences as **sexist** or **not sexist** via prompt-based methods on three open-source LLMs: Mistral v3, Llama v3.1, and Phi3-Mini.

## Approaches

### Zero-Shot Prompting
- **Original prompt** vs. **Streamlined prompt**  
- Llama 3.1 succeeded only with streamlined prompt  
- Phi3-Mini succeeded only with original prompt  
- Mistral v3 succeeded with both  

### Few-Shot Prompting
- Original prompt template with 2–14 balanced examples per query  
- Test set: 300 sentences  
- Performance:  
  - Llama 3.1 & Phi3-Mini degraded as examples ↑  
  - Mistral v3 saw slight gains  

### Intelligent Example Selection
Three retrieval methods for in-context examples:
1. **TF-IDF**  
2. **Word2Vec**  
3. **BERT embeddings**  
- BERT retrieval → marginal accuracy boost for Llama 3.1 & Phi3-Mini at high time cost  
- No benefit for Mistral v3
<!-- Table 2: Intelligent Example Selection Results :contentReference[oaicite:0]{index=0} -->

| Model           | TF-IDF Accuracy (Time) | Word2Vec Accuracy (Time) | BERT Accuracy (Time)  | Vanilla Result (Time) |
|-----------------|-----------------------:|-------------------------:|----------------------:|-----------------------:|
| Mistral (7:7)   | 68 % (9 min)           | 67 % (11 min)            | 69 % (41 min)          | 70 % (9 min)           |
| LLaMA (1:1)     | 56 % (4 min)           | 56 % (6 min)             | 58 % (36 min)          | 56 % (4 min)           |
| Phi-3 (1:1)     | 64 % (3 min)           | 62 % (5 min)             | 66 % (33 min)          | 64 % (7 min)           |
| **Average**     | 62 % (5 min)           | 61 % (7 min)             | 64 % (36 min)          | 63 % (6 min)           |


### Ensemble Methods
- **Simple voting**, **Weighted voting**, **Meta-ensemble** (logistic regression over model outputs)  
- Meta-ensemble best in dev but didn’t surpass Mistral v3 single-model accuracy  

### Prompt Tokenization Optimization
- Tokenize constant prompt sections once  
- Re-tokenize only variable parts per query  
- End-to-end latency ↓

## Experimental Setup
- **Models**: Mistral v3, Llama v3.1, Phi3-Mini  
- **Zero-Shot & Few-Shot** on 300-sentence test set  
- **Example counts**: 2, 4, …, 14  
- **Retrieval timing** measured per 300 sentences  

## Key Results

| Model       | Zero-Shot Accuracy | Few-Shot Accuracy |
|-------------|--------------------|-------------------|
| Llama 3.1   | 67 %               | 56 %              |
| Phi3-Mini   | 61 %               | 61 %              |
| Mistral v3  | 71 %               | 71 %              |

| Retrieval Method | Avg. Time (300 sents) |
|------------------|-----------------------:|
| TF-IDF           | ~5 min                 |
| Word2Vec         | ~7 min                 |
| BERT             | ~36 min                |
| Random baseline  | ~6 min                 |

## Analysis & Discussion
- **Prompt Sensitivity**: Llama 3.1 & Phi3-Mini highly sensitive to prompt format  
- **Bias**: All models biased toward “sexist” (false positives 3–10× false negatives)  
- **Dataset Complexity**: On cleaner A1 dataset → 77 % vs. fine-tuned transformer’s 83 %  

## Conclusions & Future Work
- Prompting LLMs can approach but not exceed zero-shot baselines without heavy engineering  
- **Future directions**:  
  - Tune generation parameters (temperature, beam search, `num_return_sequences`)  
  - Explore continuous prompts  
  - Integrate class-imbalance correction / debiasing strategies  

