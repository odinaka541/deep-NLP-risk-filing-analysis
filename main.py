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