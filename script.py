#

import argparse, os, json,pickle
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# nlp and ml libraries
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# deep learning libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# download required nltk data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# set random seeds
np.random.seed(42)
torch.manual_seed(42)

def load_financial_dataset():
    risk_data = {
        'operational': [
            "our business depends heavily on key personnel and their continued service",
            "disruptions to our supply chain could materially impact operations", 
            "failure of our information technology systems could harm business",
            "we face risks related to product quality and safety standards",
            "operational inefficiencies could increase costs and reduce margins",
            "we depend on third-party manufacturers for key components",
            "labor disputes or shortages could disrupt our operations",
            "our business model relies on maintaining operational excellence",
            "manufacturing defects could result in product recalls and liability",
            "we face risks from inadequate business continuity planning",
            "employee retention challenges could impact operational capability",
            "facility disruptions could interrupt production schedules",
            "vendor performance issues may affect delivery timelines",
            "process automation failures could reduce efficiency",
            "workplace safety incidents could result in operational shutdowns"
        ],
        'financial': [
            "interest rate fluctuations could increase our borrowing costs",
            "credit market conditions may limit our access to capital",
            "foreign currency exchange rate changes affect our financial results",
            "we have substantial debt obligations that require significant cash payments",
            "liquidity constraints could impact our ability to fund operations",
            "changes in accounting standards could affect reported financial results",
            "we face credit risk from customers who may not pay amounts owed",
            "our financial performance depends on effective cash management",
            "debt covenant violations could accelerate payment obligations",
            "tax law changes could increase our effective tax rate",
            "pension obligations create long-term financial commitments",
            "commodity price volatility affects our cost structure",
            "working capital requirements may strain cash resources",
            "financial reporting errors could result in restatements",
            "insurance coverage may be insufficient for potential losses"
        ],
        'market': [
            "intense competition could reduce our market share and pricing power",
            "economic downturns typically reduce demand for our products",
            "changing consumer preferences could make our products less attractive",
            "we operate in highly competitive markets with pricing pressure",
            "market saturation could limit our growth opportunities",
            "our success depends on accurately anticipating market trends",
            "economic uncertainty affects customer purchasing decisions",
            "competitive responses to our strategies could reduce effectiveness",
            "market consolidation could strengthen competitor positions",
            "cyclical demand patterns create revenue volatility",
            "new market entrants could disrupt established relationships",
            "customer concentration increases dependency risks",
            "seasonal variations affect quarterly performance",
            "brand reputation damage could impact market position",
            "distribution channel conflicts may reduce market access"
        ],
        'regulatory': [
            "changes in government regulations could increase compliance costs",
            "environmental regulations may require costly operational changes",
            "data privacy laws impose significant compliance obligations",
            "international trade policies affect our global operations",
            "healthcare regulations impact our product development and marketing",
            "antitrust laws may limit our business strategies and acquisitions",
            "financial services regulations require ongoing compliance monitoring",
            "safety regulations mandate expensive testing and certification",
            "intellectual property laws affect our ability to protect innovations",
            "tax regulations require complex compliance across multiple jurisdictions",
            "securities regulations govern our reporting and disclosure requirements",
            "labor laws affect employment practices and compensation",
            "export controls may restrict international business activities",
            "product liability regulations increase litigation exposure",
            "industry-specific regulations create compliance complexity"
        ],
        'technology': [
            "cybersecurity threats could compromise sensitive data and systems",
            "rapid technological change could make our products obsolete",
            "we face risks from software vulnerabilities and system failures",
            "data breaches could result in significant costs and reputation damage",
            "our technology infrastructure requires continuous investment and upgrades",
            "artificial intelligence and automation may disrupt our industry",
            "we depend on cloud service providers for critical operations",
            "technology integration challenges could disrupt business operations",
            "intellectual property theft could undermine competitive advantages",
            "digital transformation initiatives carry execution risks",
            "legacy system limitations may constrain business growth",
            "technology vendor dependencies create operational risks",
            "data loss incidents could damage customer relationships",
            "system scalability issues may limit business expansion",
            "emerging technologies could disrupt current business models"
        ],
        'international': [
            "political instability in key markets could disrupt operations",
            "currency devaluation in emerging markets affects revenue conversion", 
            "international trade disputes may result in tariffs or restrictions",
            "cultural differences require localized business approaches",
            "foreign regulatory requirements create compliance complexity",
            "geopolitical tensions could limit access to certain markets",
            "international expansion requires significant capital investment",
            "cross-border transactions involve foreign exchange risks",
            "we face risks from varying international legal systems",
            "global economic conditions affect our international operations",
            "sovereign debt crises could impact international markets",
            "diplomatic relations affect international business opportunities",
            "immigration policies may restrict talent acquisition",
            "international tax treaties affect global tax strategy",
            "regional conflicts could disrupt supply chains"
        ]
    }
    sentences = []
    labels = []
    companies = []
    
    company_names = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL', 'META', 'NVDA']
    
    for category, risk_list in risk_data.items():
        for sentence in risk_list:
            variations = [
                sentence,
                sentence.replace('our', 'the company\'s'),
                sentence.replace('we', 'the organization'),
                sentence.replace('could', 'may'),
                sentence.replace('would', 'might'),
            ]
            
            for variation in variations:
                sentences.append(variation)
                labels.append(category) 
                companies.append(np.random.choice(company_names))

    additional_data = [
        ("climate change regulations may increase operational costs", "regulatory"),
        ("supply chain disruptions from natural disasters pose operational risks", "operational"),
        ("rising inflation could increase our cost structure", "financial"),
        ("changing demographics affect our target market", "market"),
        ("quantum computing advances may threaten current encryption", "technology"),
        ("brexit impacts our european operations", "international"),
        ("machine learning algorithms require continuous model updates", "technology"),
        ("trade war escalations could affect international revenues", "international"),
        ("pension fund obligations create long-term financial liabilities", "financial"),
        ("customer acquisition costs continue to increase in competitive markets", "market"),
        ("regulatory compliance requires specialized legal expertise", "regulatory"),
        ("manufacturing capacity constraints limit growth potential", "operational"),
        ("intellectual property disputes may result in licensing costs", "regulatory"),
        ("cloud infrastructure outages could disrupt customer service", "technology"),
        ("currency hedging strategies may not fully mitigate exchange rate risks", "international"),
        ("working capital management affects short-term liquidity", "financial"),
        ("market research indicates shifting consumer preferences", "market"),
        ("equipment maintenance schedules affect production efficiency", "operational"),
    ]
    
    for sentence, label in additional_data:
        sentences.append(sentence)
        labels.append(label)
        companies.append(np.random.choice(company_names))
    
    df = pd.DataFrame({
        'sentence': sentences,
        'category': labels,
        'company': companies
    })
    
    df = shuffle(df, random_state=42).reset_index(drop=True)
    
    print(f"loaded dataset with {len(df)} samples")
    print(f"categories: {df['category'].value_counts().to_dict()}")
    
    return df

class RiskAnalysisDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class TransformerRiskClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=6, dropout=0.3):
        super(TransformerRiskClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

def train_transformer_model(model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
        
        model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        val_accuracy = total_correct / total_samples
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'epoch {epoch+1}/{num_epochs}:')
        print(f'  average training loss: {avg_train_loss:.4f}')
        print(f'  validation accuracy: {val_accuracy:.4f}')
    
    return train_losses, val_accuracies

def evaluate_model(model, test_loader):
    """evaluate model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

def run_analysis(output_dir='results', num_epochs=3):
    os.makedirs(output_dir, exist_ok=True)
    
    print("starting financial risk factor analysis")
    print("loading financial risk dataset...")
    
    df = load_financial_dataset()
    
    if len(df) == 0:
        print("no data loaded")
        return None
    
    df.to_csv(os.path.join(output_dir, 'risk_sentences.csv'), index=False)
    
    print("preparing data for transformer model...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df['category'])
    
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['sentence'].tolist(), encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"train samples: {len(X_train)}")
    print(f"validation samples: {len(X_val)}")
    print(f"test samples: {len(X_test)}")
    
    train_dataset = RiskAnalysisDataset(X_train, y_train, tokenizer)
    val_dataset = RiskAnalysisDataset(X_val, y_val, tokenizer)
    test_dataset = RiskAnalysisDataset(X_test, y_test, tokenizer)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(label_encoder.classes_)
    model = TransformerRiskClassifier(num_classes=num_classes)
    
    print(f"training transformer model with {num_classes} classes")
    print(f"classes: {label_encoder.classes_}")
    
    train_losses, val_accuracies = train_transformer_model(
        model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=2e-5
    )
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'risk_classifier_model.pth'))
    
    predictions, true_labels = evaluate_model(model, test_loader)
    
    accuracy = accuracy_score(true_labels, predictions)
    print(f"test accuracy: {accuracy:.4f}")
    
    results = {
        'accuracy': float(accuracy),
        'num_classes': int(num_classes),
        'total_sentences': len(df),
        'train_losses': [float(x) for x in train_losses],
        'val_accuracies': [float(x) for x in val_accuracies],
        'class_names': label_encoder.classes_.tolist(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    class_names = label_encoder.classes_
    report = classification_report(true_labels, predictions, target_names=class_names, output_dict=True)
    
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print("classification report:")
    print(classification_report(true_labels, predictions, target_names=class_names))
    
    create_visualizations(df, train_losses, val_accuracies, true_labels, predictions, 
                         class_names, output_dir)
    
    save_example_predictions(X_test, true_labels, predictions, class_names, output_dir)
    
    print(f"analysis complete! results saved to {output_dir}")
    print(f"model achieved {accuracy:.1%} accuracy on risk factor classification")
    
    return results

def create_visualizations(df, train_losses, val_accuracies, true_labels, predictions, 
                         class_names, output_dir):
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 10))
    
    # training history
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('training loss', fontsize=12, fontweight='bold')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(val_accuracies, 'g-', linewidth=2)
    plt.title('validation accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid(True, alpha=0.3)
    
    # CMx
    plt.subplot(2, 3, 3)
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('confusion matrix', fontsize=12, fontweight='bold')
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # category distribution
    plt.subplot(2, 3, 4)
    df['category'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('risk category distribution', fontsize=12, fontweight='bold')
    plt.xlabel('category')
    plt.ylabel('count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # company distribution
    plt.subplot(2, 3, 5)
    df['company'].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title('sentences by company', fontsize=12, fontweight='bold')
    plt.xlabel('company')
    plt.ylabel('count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # performance comparison
    plt.subplot(2, 3, 6)
    accuracy = accuracy_score(true_labels, predictions)
    baseline_acc = max(np.bincount(true_labels)) / len(true_labels)
    
    metrics = ['bert model', 'baseline']
    values = [accuracy, baseline_acc]
    colors = ['green', 'red']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    plt.title('model performance', fontsize=12, fontweight='bold')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'analysis_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"visualizations saved to {output_dir}/analysis_plots.png")

def save_example_predictions(X_test, true_labels, predictions, class_names, output_dir):
    examples = []
    
    for i in range(min(20, len(X_test))):
        sentence = X_test[i][:300] + "..." if len(X_test[i]) > 300 else X_test[i]
        true_label = class_names[true_labels[i]]
        pred_label = class_names[predictions[i]]
        correct = true_label == pred_label
        
        examples.append({
            'sentence': sentence,
            'true_category': true_label,
            'predicted_category': pred_label,
            'correct': correct
        })
    
    examples_df = pd.DataFrame(examples)
    examples_df.to_csv(os.path.join(output_dir, 'example_predictions.csv'), index=False)
    
    print(f"example predictions saved to {output_dir}/example_predictions.csv")

def parse_arguments():
    parser = argparse.ArgumentParser(description='financial risk factor analysis using bert')
    
    parser.add_argument('--epochs', type=int, default=3,
                       help='number of training epochs')
    
    parser.add_argument('--output', type=str, default='results',
                       help='output directory for results')
    
    parser.add_argument('--batch-size', type=int, default=8,
                       help='batch size for training')
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        
        results = run_analysis(
            output_dir=args.output,
            num_epochs=args.epochs
        )
        
        if results:
            print("analysis completed successfully")
            print(f"final accuracy: {results['accuracy']:.1%}")
        else:
            print("analysis failed")
            exit(1)
            
    except KeyboardInterrupt:
        print("analysis interrupted by user")
        exit(0)
    except Exception as e:
        print(f"analysis failed: {str(e)}")
        exit(1)