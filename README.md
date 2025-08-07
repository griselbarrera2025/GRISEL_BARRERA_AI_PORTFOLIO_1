Grisel-Barrera-HCC-AI/
├── README.md
├── NLP-ITAI2373/
│   ├── Text-Processing-Project/
│   │   ├── preprocessing.ipynb
│   │   ├── README.md
│   │   ├── sample_data/
│   │   └── outputs/
│   └── Sentiment-And-Emotion-Analysis/
│       ├── classifier.py
│       ├── README.md
│       └── results/
│   ├── Text-Representation/
│   │   ├── tfidf_bow_embeddings.ipynb      # Code for BoW, TF-IDF, Word Embeddings
│   │   ├── README.md                       # Based on the Text Representation README I wrote
│   │   ├── sample_data/
│   │   └── outputs/
│   ├── Intro-to-Audio-and-Preprocessing/
│   │   ├── audio_preprocessing.ipynb       # Code for audio loading, cleaning, MFCCs
│   │   ├── README.md
│   │   ├── sample_audio/
│   │   └── outputs/
│   ├── Sentiment-and-Emotion-Analysis/
│   │   ├── sentiment_emotion.py            # Combined sentiment + emotion model
│   │   ├── README.md
│   │   └── results/
│   ├── Syntax-Parsing-and-Semantic-Analysis/
│   │   ├── syntax_semantic.py              # POS tagging, parsing, NER
│   │   ├── README.md
│   │   └── outputs/
│   └── Part-of-Speech-Tagging/
│       ├── classifier.py
│       ├── README.md
│       └── results/
├── Text-Classification-and-NER/
│   └── SmartSensorSystem/
│       ├── iot_model.py
│       ├── README.md
│       └── sensor_data/
└── Presentation/
    └── Pf_GriselBarrera_ITAI2376.pdf


# Grisel Barrera – Applied AI & Robotics Portfolio

Welcome to my professional AI portfolio, showcasing hands-on projects completed during the Applied AI and Robotics program at Houston Community College.

## 👩‍🎓 About Me
I am currently enrolled in the Applied AI & Robotics program at HCC, where I’ve developed skills in deep learning, NLP, computer vision, edge AI, and conversational AI systems.

## 📘 Courses & Skills
- **Deep Learning (ITAI 2376)** – CNNs, RNNs, GANs, U-Net, optimization, diffusion models
- **Natural Language Processing (ITAI 2373)** – Text preprocessing, sentiment analysis, NER, POS tagging, topic modeling
- **AI at the Edge / IoT (ITAI 3377)** – Embedded AI systems, low-latency models, sensor data integration
- **Conversational AI** – Dialogue systems, intent classification, and LLMs

## 📌 Featured Projects
-  [BBC News Classification (NLP)](./NLP-ITAI2373/Text-Processing-Project/)
-  [Emotion Detection from Text](./NLP-ITAI2373/Emotion-Classifier/)
-  [Text_Representation_with_Bag-of-Words, TF-IDF, and Word Embeddings]_(NLP_ITAI2373/TF_IDF)
-  [Smart IoT Sensor Alert System](./AI-at-the-Edge-IoT-ITAI3377/SmartSensorSystem/)
-  [Intro_To_Audio_and Preprocessing]_(/NLP_ITAI2373/Intro_to_Audio_&_Preprocessing/)
-  [Sentiment_and_Emotion_Analysis]_(NLP_ITAI2373/Sentiment_and_Emotion_Analysis/)
-  [Syntax Parsing & Semantic Analysis]_(NLP_ITAI/Syntax_Parsing_&_Semantic_Analysis/)
  
## 📬 Contact
📧 Email: griselbarrera2016@gmail.com  
📍 Location: Houston, TX  
📎 LinkedIn: [linkedin.com/in/GriselBarrera]
# BBC News Classification with NLP

## 🧠 Problem Statement
Classify news articles into categories such as business, politics, tech, etc., using NLP techniques on the BBC News dataset.

## 🔧 Approach and Methodology
- Text cleaning (lowercasing, punctuation removal, stopwords, lemmatization)
- TF-IDF vectorization
- Logistic Regression, Random Forest, and Naive Bayes models
- Evaluation with F1 score and confusion matrix

## 📈 Results
- Best model: Logistic Regression with 91.2% accuracy
- Clear separation in confusion matrix between categories like sport and tech
- Balanced performance across all five categories

## ✅ Learning Outcomes
- Mastery of preprocessing and feature extraction pipelines
- Gained hands-on experience with multi-class classification and model evaluation
- Understood limitations of classical models vs. neural approaches

## 📦 Requirements
```bash
pip install -r requirements.txt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load sample data
df = pd.DataFrame({"text": ["This is a sample sentence.", "Another example goes here."]})
df['cleaned'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])

print("TF-IDF Matrix Shape:", X.shape)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample emotion-labeled dataset
data = {
    'text': ['I love this!', 'I am so angry', 'This is depressing', 'I am excited!', 'This is sad'],
    'emotion': ['joy', 'anger', 'sadness', 'joy', 'sadness']
}
df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Text Representation with Bag-of-Words, TF-IDF, and Word Embeddings

## 🧠 Problem Statement
Explore different ways to convert raw text into numerical features for machine learning models. This project compares the effectiveness of Bag-of-Words (BoW), TF-IDF, and Word Embeddings.

## 🔧 Approach and Methodology
- Preprocessed text using tokenization, stop word removal, and lemmatization.
- Implemented three methods of vectorization:
  - **Bag-of-Words** using `CountVectorizer`
  - **TF-IDF** using `TfidfVectorizer`
  - **Word Embeddings** using pre-trained GloVe vectors

## 📈 Results and Evaluation
- Compared vector shapes and vocabulary size
- Demonstrated how dense vs sparse vectors affect ML models
- TF-IDF captured term importance better than BoW
- Word embeddings captured semantic similarity

## ✅ Learning Outcomes
- Gained practical experience with feature engineering in NLP
- Understood limitations and advantages of different vectorization techniques
- Learned how to integrate external word embedding models

## 📦 Requirements
```bash
pip install pandas scikit-learn nltk gensim


---

### 📁 **Syntax Parsing & Semantic Analysis**  
**File:** `README.md`

```markdown
# Syntax Parsing and Semantic Analysis in NLP

## 🧠 Problem Statement
Analyze the grammatical structure and meaning of text using syntax trees and semantic role labeling.

## 🔧 Approach and Methodology
- Used **SpaCy** for:
  - Part-of-Speech (POS) Tagging
  - Dependency Parsing
  - Named Entity Recognition
- Explored tree structures and grammatical relations
- Conducted semantic analysis using SpaCy's token attributes:
  - `.lemma_`, `.dep_`, `.head`, `.ent_type_`, etc.

## 📈 Results and Evaluation
- Visualized syntactic trees and dependency arcs
- Identified subject-verb-object relationships
- Extracted entities and their semantic roles

## ✅ Learning Outcomes
- Learned how modern NLP pipelines analyze grammar
- Applied semantic analysis to better understand sentence meaning
- Practiced using token-level NLP tools programmatically

## 📦 Requirements
```bash
pip install spacy
python -m spacy download en_core_web_sm





