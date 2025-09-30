import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import re

# 1. Incarcarea datelor si explorarea acestora

df = pd.read_csv('data/IMLP4_TASK_03-products.csv')
print('Shape:', df.shape)
print(df.head())
print(df.info())

# 2. Pregatirea si curatarea datelor

# Normalizam denumirile coloanelor
def normalize_column_name(col_name):
    # elimina spatii la inceput si sfarsit
    col_name = col_name.strip()
    # transforma in lowercase
    col_name = col_name.lower()
    # elimina caractere speciale, pastrand doar litere, cifre si underscore
    col_name = ''.join(c if c.isalnum() else '_' for c in col_name)
    # elimina underscore-uri multiple
    while '__' in col_name:
        col_name = col_name.replace('__', '_')
    # elimina underscore la inceput si sfarsit
    col_name = col_name.strip('_')
    return col_name

# Aplicam normalizarea
df.columns = [normalize_column_name(col) for col in df.columns]

# Verificam valorile lipsa si eliminam randurile cu valori lipsa
df = df.dropna()

# Verificam valorile duplicate si eliminam duplicatele
df = df.drop_duplicates()

# Standardizam textul din coloana 'product_title'
df['product_title'] = (
    df['product_title']
    .astype(str)              # fortam text
    .str.strip()              # eliminam spatiile la inceput/sfarsit
    .str.lower()              # trecem tot in litere mici
)

df['category_label'] = df['category_label'].replace({
    'Mobile Phone': 'Mobile Phones',
    'fridge': 'Fridges',
    'CPU': 'CPUs'
})

# Eliminam caracterele speciale, pastrand doar litere, cifre si spatii
df['product_title'] = df['product_title'].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', x))

print('Shape after cleaning:', df.shape)

print(df.head())

# Ingineria caracteristicilor
def numeric_features_from_titles(titles_series):
    titles = titles_series.fillna("").astype(str)
    num_words = titles.str.split().apply(len).values.reshape(-1,1)
    num_chars = titles.str.len().values.reshape(-1,1)
    num_digits = titles.str.count(r'\d').values.reshape(-1,1)
    has_upper_acronym = titles.str.contains(r'\b[A-Z]{2,}\b').astype(int).values.reshape(-1,1)
    has_slash = titles.str.contains(r'[/\\-]').astype(int).values.reshape(-1,1)
    return np.hstack([num_words, num_chars, num_digits, has_upper_acronym, has_slash])

# Impartirea datelor in seturi de antrenament si test

X = df['product_title']
y = df['category_label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Definim modelele
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Linear SVC": LinearSVC()
}

# Rulam cross-validation pentru fiecare model
results = {}
for name, model in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("clf", model)
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    results[name] = (scores.mean(), scores.std())

# Afisam rezultatele
for model, (mean_acc, std_acc) in results.items():
    print(f"{model}: {mean_acc:.4f} (+/- {std_acc:.4f})")

# Antrenam modelul final (Linear SVC a avut cele mai bune rezultate)
final_model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LinearSVC())  # poti schimba cu LogisticRegression(max_iter=1000) daca preferi
])

# Antrenam pe TOT setul de antrenament
final_model.fit(X_train, y_train)

# Evaluare pe setul de test
y_pred = final_model.predict(X_test)
print("\n=== Evaluare pe test set ===")
print(classification_report(y_test, y_pred, zero_division=0))

# Salvam modelul final pentru utilizare ulterioara
joblib.dump(final_model, "final_product_classifier.pkl")
print("Modelul final a fost salvat in 'final_product_classifier.pkl'")
