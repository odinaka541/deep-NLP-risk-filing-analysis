# imports
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
from transformers import AutoTokenizer, AutoModel, AdamW
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
        
        for entry in soup.find_all('entry'):
            filing_href = entry.find('filing-href')
            filing_date = entry.find('filing-date')
            
            if filing_href and filing_date:
                filings.append({
                    'url': filing_href.text,
                    'date': filing_date.text
                })
        
        return filings
    
    def extract_document_url(self, filing_url):
        try:
            response = requests.get(filing_url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # looking for the main document link
            for link in soup.find_all('a'):
                if link.get('href') and '10-k' in link.get('href', '').lower():
                    doc_url = urljoin(self.base_url, link['href'])
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
