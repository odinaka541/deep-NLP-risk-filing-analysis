# Financial Risk Factor Analysis from SEC filings (using DL and NLP)

BERT-based classifier for categorizing financial risk statements from 10-K filings and corporate documents.

## Installation

```bash
pip install torch transformers sklearn pandas matplotlib seaborn nltk
```

## Usage

Basic execution:
```bash
python script.py
```

With custom parameters:
```bash
python script.py --epochs 5 --output results/
```

## Risk Categories

- **Operational**: Supply chain, personnel, manufacturing
- **Financial**: Money, credit, liquidity, interest rates, debt
- **Market**: Competition, demand, economic conditions
- **Regulatory**: Compliance, legal changes, policy
- **Technology**: Cybersecurity, system failures, IT risks
- **International**: Geopolitical, currency, trade risks

## Output Files

- `risk_sentences.csv` - Input dataset
- `risk_classifier_model.pth` - Trained BERT model
- `results.json` - Performance metrics
- `classification_report.json` - Detailed classification results
- `predictions.csv` - Sample predictions
- `analysis_plots.png` - Performance visualizations

## Data

Uses synthetically generated financial risk statements based on common 10-K filing language patterns. Includes data augmentation through sentence variations.

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Scikit-learn
- NLTK