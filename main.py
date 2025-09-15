# imports
import argparse, os, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import warnings
warnings.filterwarnings('ignore')

# nlp and ml libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# deep learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
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
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# set random seeds
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------

class SECDataCollector:
    def __init__(self):
        """collect and parse sec 10-k filings from edgar database"""
        self.base_url = "https://www.sec.gov"
        self.headers = {
            'User-Agent': 'research-project contact@university.edu'
        }
    
    def get_company_filings(self, cik, form_type='10-K', count=5):
        """
        get recent filings for a company using
            cik: company central index key
            form_type: type of filing to retrieve
            count: number of recent filings to get
        """
        # cik to 10 digits
        cik_formatted = str(cik).zfill(10)
        
        # searching for filings
        search_url = f"{self.base_url}/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'CIK': cik_formatted,
            'type': form_type,
            'dateb': '',
            'owner': 'exclude',
            'count': count,
            'output': 'xml'
        }
        
        try:
            response = requests.get(search_url, params=params, headers=self.headers)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"error fetching filings for cik {cik}: {e}")
            return None

    def parse_filing_links(self, xml_content):
        """extracting document links from sec xml response"""
        soup = BeautifulSoup(xml_content, 'xml')
        filings = []
        
        results = soup.find('results')
        if results:
            for filing in results.find_all('filing'):
                filing_href = filing.find('filingHREF')
                filing_date = filing.find('dateFiled') 
                form_type = filing.find('type')
                
                if filing_href and filing_date and form_type:
                    if '10-K' in form_type.text:
                        filings.append({
                            'url': filing_href.text,
                            'date': filing_date.text
                        })
                        print(f"  Found 10-K filing from {filing_date.text}")
        
        print(f"  Total filings parsed: {len(filings)}")
        return filings
        
    def extract_document_url(self, filing_url):
        try:
            response = requests.get(filing_url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for link in soup.find_all('a'):
                href = link.get('href', '')
                link_text = link.get_text().lower()
                
                if href and not any(skip in href.lower() for skip in ['exhibit', 'ex-', 'graphic']):
                    if any(indicator in href.lower() for indicator in ['10-k', '10k']) or \
                    any(indicator in link_text for indicator in ['10-k', 'annual report']):
                        doc_url = urljoin(self.base_url, href)
                        print(f"  Selected document: {href}")
                        return doc_url
            
            print("  No clear main document found, trying fallback...")
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href and '.htm' in href and not any(skip in href.lower() for skip in ['exhibit', 'ex-', 'graphic']):
                    doc_url = urljoin(self.base_url, href)
                    print(f"  Fallback document: {href}")
                    return doc_url
                        
        except Exception as e:
            print(f"error extracting document url: {e}")
            
        return None
    
    def download_document(self, doc_url):
        try:
            time.sleep(0.1)  # rate limiting
            response = requests.get(doc_url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"error downloading document: {e}")
            return None

#
class DocumentProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def extract_risk_factors_section(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # common patterns for risk factors section
        risk_patterns = [
            r'item\s*1a\s*[\.\-\s]*risk\s*factors',
            r'risk\s*factors',
            r'item\s*1a',
        ]
        
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).lower()
        
        # finding risk factors section
        for pattern in risk_patterns:
            match = re.search(pattern, text)
            if match:
                start_pos = match.start()
                
                end_patterns = [
                    r'item\s*1b',
                    r'item\s*2\s*[\.\-\s]*properties',
                    r'item\s*2[^a-z]'
                ]
                
                end_pos = len(text)
                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, text[start_pos:])
                    if end_match:
                        end_pos = start_pos + end_match.start()
                        break
                
                risk_section = text[start_pos:end_pos]
                return self.clean_text(risk_section)
        
        return None
    
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'page\s+\d+', '', text)
        text = re.sub(r'table\s+of\s+contents', '', text)

        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text) # special chars
        
        return text.strip()
    
    def extract_risk_sentences(self, text):
        sentences = sent_tokenize(text)
        
        # filterinh for sentences that likely contain risk information
        risk_keywords = [
            'risk', 'risks', 'uncertain', 'uncertainty', 'may', 'could', 
            'might', 'potential', 'adverse', 'materially', 'significant',
            'impact', 'affect', 'failure', 'inability', 'depends', 'dependent'
        ]
        
        risk_sentences = []
        for sentence in sentences:
            if len(sentence.split()) > 10:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in risk_keywords):
                    risk_sentences.append(sentence)
        
        return risk_sentences

#
class RiskClassifier:
    def __init__(self):
        self.risk_categories = {
            'operational': [
                'operations', 'operational', 'business', 'management', 'employees',
                'key personnel', 'technology', 'systems', 'processes', 'supply chain',
                'manufacturing', 'production', 'quality', 'safety'
            ],
            'financial': [
                'financial', 'cash', 'liquidity', 'debt', 'credit', 'capital',
                'funding', 'cash flow', 'revenue', 'profitability', 'costs',
                'expenses', 'tax', 'accounting', 'auditing'
            ],
            'market': [
                'market', 'competition', 'competitive', 'competitors', 'demand',
                'customer', 'customers', 'pricing', 'economic', 'economy',
                'recession', 'inflation', 'interest rates'
            ],
            'regulatory': [
                'regulation', 'regulatory', 'compliance', 'legal', 'law', 'laws',
                'government', 'policy', 'policies', 'sec', 'fda', 'environmental',
                'litigation', 'lawsuit', 'patent', 'intellectual property'
            ],
            'technology': [
                'technology', 'technological', 'cyber', 'cybersecurity', 'data',
                'information', 'security', 'breach', 'hacking', 'software',
                'hardware', 'internet', 'digital', 'innovation'
            ],
            'international': [
                'international', 'global', 'foreign', 'overseas', 'currency',
                'exchange', 'trade', 'tariff', 'political', 'geopolitical',
                'country', 'countries', 'region', 'regional'
            ]
        }
    
    def classify_risk_sentence(self, sentence):
        sentence_lower = sentence.lower()
        
        category_scores = {}
        for category, keywords in self.risk_categories.items():
            score = sum(1 for keyword in keywords if keyword in sentence_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            # rturing the  category with highest score
            return max(category_scores, key=category_scores.get)
        else:
            return 'other'

# transformer-based model for risk analysis
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
    def __init__(self, model_name='bert-base-uncased', num_classes=7, dropout=0.3):
        super(TransformerRiskClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)


# !!!!!!!!!!!!!!!!!!!!!!!
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
        
        # val
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

# !!!!!!! main analysis pipeline
def run_analysis(companies_dict, output_dir='results', num_epochs=3, max_sentences_per_company=50):

    os.makedirs(output_dir, exist_ok=True)
    
    print("starting sec filing risk factor analysis")
    print(f"analyzing companies: {list(companies_dict.keys())}")
    
    sec_collector = SECDataCollector()
    doc_processor = DocumentProcessor()
    risk_classifier = RiskClassifier()
    
    all_risk_sentences = []
    all_labels = []
    all_companies = []
    
    print("collecting and processing sec filings...")
    
    for company, cik in companies_dict.items():
        print(f"\nPROCESSING {company} (cik: {cik})")
        
        # D !!!!!
        filings_xml = sec_collector.get_company_filings(cik, count=2)
        if not filings_xml:
            print(f" !!! NO XML RESPONSE for {company}")
            continue
        else:
            print(f"!!! Got XML response ({len(filings_xml)} chars)")
            print(f"First 200 chars: {filings_xml[:200]}")
            
        # D
        filings = sec_collector.parse_filing_links(filings_xml)
        print(f"!!! Found {len(filings)} filings")
        if filings:
            print(f"Latest filing: {filings[0]}")
        else:
            print("!!! No filings parsed from XML")
            continue
        
        # Process first filing
        for filing in filings[:1]:
            print(f"\n--- Processing filing from {filing['date']} ---")
            print(f"Filing URL: {filing['url']}")
            
            # DEBUG: Extract document URL
            doc_url = sec_collector.extract_document_url(filing['url'])
            if not doc_url:
                print(f"!!! Could not extract document URL")
                continue
            else:
                print(f"!!! Document URL: {doc_url}")
            
            # DEBUG: Download document
            html_content = sec_collector.download_document(doc_url)
            if not html_content:
                print(f"!!! Could not download document")
                continue
            else:
                print(f"!!! Downloaded document ({len(html_content)} chars)")
                
            # !!! Check for risk content
            if 'risk' in html_content.lower():
                print(f"!!! 'risk' found in document")
                risk_count = html_content.lower().count('risk')
                print(f"'risk' appears {risk_count} times")
            else:
                print(f"!!! No 'risk' found in document")
                continue
            
            # DEBUG: Extract risk section
            risk_text = doc_processor.extract_risk_factors_section(html_content)
            if not risk_text:
                print(f"!!! Could not extract risk factors section")
                # Let's see what we can find
                soup = BeautifulSoup(html_content, 'html.parser')
                text = soup.get_text().lower()
                if 'item 1a' in text:
                    print("Found 'item 1a' in text")
                if 'risk factors' in text:
                    print("Found 'risk factors' in text")
                continue
            else:
                print(f"!!! Extracted risk section ({len(risk_text)} chars)")
                print(f"First 300 chars: {risk_text[:300]}")
            
            # !!! Extract sentences
            sentences = doc_processor.extract_risk_sentences(risk_text)
            print(f"!!! Extracted {len(sentences)} risk sentences")
            
            if sentences:
                print(f"First sentence: {sentences[0][:200]}...")
                
                for sentence in sentences[:max_sentences_per_company]:
                    category = risk_classifier.classify_risk_sentence(sentence)
                    all_risk_sentences.append(sentence)
                    all_labels.append(category)
                    all_companies.append(company)
                    
                print(f"Added {min(len(sentences), max_sentences_per_company)} sentences")
            else:
                print("!!! No risk sentences extracted from risk section")
    
    print(f"\n results: g")
    print(f"Total risk sentences collected: {len(all_risk_sentences)}")
    
    if not all_risk_sentences:
        print("!!! NO RISK SENTENCES EXTRACTED - CHECK DEBUG OUTPUT ABOVE")
        return None
    

    df = pd.DataFrame({
        'sentence': all_risk_sentences,
        'category': all_labels,
        'company': all_companies
    })
    
    print(f"total sentences collected: {len(df)}")
    print(f"category distribution:\n{df['category'].value_counts()}")
    
    df.to_csv(os.path.join(output_dir, 'risk_sentences.csv'), index=False)
    
    print("preparing data for transformer model...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df['category'])
    
    import pickle
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # splitting data
    X_train, X_test, y_train, y_test = train_test_split(
        df['sentence'].tolist(), encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"train samples: {len(X_train)}")
    print(f"validation samples: {len(X_val)}")
    print(f"test samples: {len(X_test)}")
    
    # 
    train_dataset = RiskAnalysisDataset(X_train, y_train, tokenizer)
    val_dataset = RiskAnalysisDataset(X_val, y_val, tokenizer)
    test_dataset = RiskAnalysisDataset(X_test, y_test, tokenizer)

    batch_size = 8 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # model init and train
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
    
    # saving results !!!!!!!!!!!!
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
    
    plt.subplot(2, 3, 6)
    accuracy = accuracy_score(true_labels, predictions)
    baseline_acc = max(np.bincount(true_labels)) / len(true_labels)
    
    metrics = ['model accuracy', 'baseline accuracy']
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
    parser = argparse.ArgumentParser(description='sec filing risk factor analysis')
    
    parser.add_argument('--companies', type=str, 
                       default='AAPL,MSFT,TSLA,AMZN,GOOGL',
                       help='comma-separated list of company symbols')
    
    parser.add_argument('--epochs', type=int, default=3,
                       help='number of training epochs')
    
    parser.add_argument('--output', type=str, default='results',
                       help='output directory for results')
    
    parser.add_argument('--max-sentences', type=int, default=50,
                       help='maximum sentences per company')
    
    parser.add_argument('--batch-size', type=int, default=8,
                       help='batch size for training')
    
    return parser.parse_args()

def get_company_cik_mapping():
    """ display proper company symbols"""
    return {
        'AAPL': '320193',      # apple 
        'MSFT': '789019',      # microsoft 
        'AMZN': '1018724',     # amazon 
        'GOOGL': '1652044',    # alphabet 
        'TSLA': '1318605',     # tesla 
        'META': '1326801',     # meta
        'NVDA': '1045810',     # nvidia 
        'JPM': '19617',        # jpmorgan 
        'JNJ': '200406',       # johnson & johnson
        'V': '1403161'         # visa inc
    }

if __name__ == "__main__":
    try:
        args = parse_arguments()
        
        cik_mapping = get_company_cik_mapping()
        
        company_symbols = [c.strip().upper() for c in args.companies.split(',')]
        companies_dict = {}
        
        for symbol in company_symbols:
            if symbol in cik_mapping:
                companies_dict[symbol] = cik_mapping[symbol]
            else:
                print(f"cik not found for {symbol}, skipping")
        
        if not companies_dict:
            print("no valid companies found")
            exit(1)
        
        results = run_analysis(
            companies_dict=companies_dict,
            output_dir=args.output,
            num_epochs=args.epochs,
            max_sentences_per_company=args.max_sentences
        )
        
        if results:
            print("analysis completed successfully")
        else:
            print("analysis failed")
            exit(1)
            
    except KeyboardInterrupt:
        print("analysis interrupted by user")
        exit(0)
    except Exception as e:
        print(f"analysis failed: {str(e)}")
        exit(1)