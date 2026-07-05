# NLP From Scratch
 
Foundational natural language processing techniques implemented from primitives rather than pulled from high-level libraries. Two projects in this repository:
 
1. **`language-models/`** — Multilingual unigram and n-gram language models in English and Japanese, applied to token-probability estimation and manuscript recovery.
2. **`pos-tagging-hmm/`** — A Hidden Markov Model part-of-speech tagger built with Viterbi decoding, comparing three smoothing schemes on the Brown corpus.
## Why "from scratch"?
 
Modern NLP is dominated by transformer-based pretrained models, and calling `spaCy` or `transformers` will get you competent results on almost any task. That's the right choice for production, but it hides the underlying mechanics. This repository is about the primitives: what a probability distribution over tokens actually looks like when you compute one by hand, why unsmoothed maximum-likelihood estimation breaks on unseen words, how the Viterbi algorithm avoids the exponential brute-force search that naive decoding would require, and how tokenization strategy has to change when your language doesn't put spaces between words. Those primitives underlie everything above them in the stack.
 
---
 
# Project 1: Multilingual Language Models
 
**File:** `language-models/nlp-language-models.ipynb`
 
Unigram and n-gram language models built from scratch across English and Japanese, applied to three related language-modeling tasks:
 
1. **English unigram model** on Bram Stoker's *Dracula* — computing per-token probabilities from cleaned tokenized text.
2. **Japanese unigram model** on E. Phillips Oppenheim's *The Great Impersonation* (Japanese edition) — using the `janome` morphological analyzer to segment Japanese text without whitespace boundaries.
3. **English n-gram model** on Shakespeare's corpus — bigram and fivegram models applied to a manuscript-recovery task, predicting 1,740 tokens that have been replaced with `<DELETED>` placeholders, scored against a validation reference.
### The multilingual challenge
 
Japanese doesn't put spaces between words, so `word_tokenize()` doesn't apply. The `janome` morphological analyzer segments the text into constituent morphemes; a separate `is_japanese()` helper filters to hiragana / katakana / kanji tokens only, excluding Latin characters and numerals that appear in the mixed source. The probability math stays identical to English — the demonstration is that language-modeling *concepts* transfer even when the implementation doesn't.
 
### The n-gram recovery task
 
Longer contexts give the fivegram model more disambiguating information, but they also mean more sparse contexts where no matching n-gram exists in training data. The bigram-vs-fivegram comparison in this notebook makes the classic n-gram sparsity tradeoff concrete on a real recovery task.
 
### Tech Stack
 
- Python 3
- `nltk` — corpus loading, tokenization, `FreqDist`, `ConditionalFreqDist`, `ConditionalProbDist`
- `janome` — Japanese morphological analysis
- `pickle` — model persistence
---
 
# Project 2: HMM POS Tagger with Viterbi Decoding
 
**File:** `pos-tagging-hmm/hmm-viterbi-pos-tagger.ipynb`
 
A Hidden Markov Model trained on the Brown corpus for part-of-speech tagging, using the Viterbi algorithm to find the most probable tag sequence. Compares three smoothing schemes — MLE (no smoothing), Expected Likelihood Estimation, and Lidstone smoothing (α = 0.01) — to examine how the choice of probability estimator affects tagging accuracy on unseen words.
 
### Dataset
 
The **Brown corpus** (1961, Brown University — the first million-word electronic corpus of English), categories `news`, `editorial`, `reviews`, with the universal POS tagset (12 tags).
 
### Approach
 
1. Build vocabulary and tag inventories from the training set.
2. Train three HMM variants with different smoothing estimators.
3. Decode with Viterbi via NLTK's `best_path()`.
4. Evaluate on overall token accuracy, sentence-level accuracy, per-tag precision / recall / F1, and confusion matrix.
5. Serialize the best model with `dill` (which handles NLTK's nested objects that stock `pickle` can't).
### Results (MLE variant)
 
- Total tokens: 40,434 — correct: **51.28%**
- Sentences with every tag correct: 489 / 1,875 (**26.08%**)
- Precision high (0.90+) on real tags, recall around 0.50 — the classic HMM-without-smoothing failure mode where unseen words all collapse into the `X` catch-all tag.
State-of-the-art POS tagging hits ~97% token accuracy; the gap between that and 51% is largely the smoothing story plus modern neural approaches. This project's value isn't the accuracy number — it's implementing transition and emission probabilities directly and understanding *why* the unsmoothed baseline fails on real data.
 
### Tech Stack
 
- Python 3
- `nltk` — Brown corpus, `HiddenMarkovModelTrainer`, `LidstoneProbDist`, `MLEProbDist`, `ELEProbDist`
- `scikit-learn` — train/test split and metrics
- `dill` — model serialization
---
## How to Run
 
```bash
pip install nltk janome scikit-learn dill
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('brown'); nltk.download('universal_tagset')"
jupyter notebook
```
 
## Data Sources
 
- *Dracula* by Bram Stoker — Project Gutenberg ebook 345.
- *The Great Impersonation* (Japanese translation) by E. Phillips Oppenheim — Project Gutenberg.
- Shakespeare training and test files — equivalent public-domain Shakespeare corpora are available on Project Gutenberg.
- Brown corpus — accessed via `nltk.download('brown')`.
