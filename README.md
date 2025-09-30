# Product Classification Project

Acest proiect antreneaza si testeaza un model de clasificare a produselor pe baza titlurilor lor folosind Python si scikit-learn.

## Structura proiectului

project/
 data/ IMLP4_TASK_03-products.csv # Dataset-ul initial
 train_model.py # Script pentru antrenarea si salvarea modelului
 predict_category.py # Script pentru prezicerea categoriei unui titlu
 final_product_classifier.pkl # Modelul salvat (generat de train_model.py)
 product_classification_analysis.ipynb # Analiza ipynb
 README.md # Documentatia proiectului

## Cerinte

- Python 3.8+
- Biblioteci Python:
    pandas numpy scikit-learn joblib

## Antrenarea modelului
Asigura-te ca fisierul IMLP4_TASK_03-products.csv se afla in folderul data/.

Ruleaza scriptul Train_model.py:
python Train_model.py

## Scriptul va:

- Curata si normaliza datele
- Imparti setul in train/test
- Testa mai multe modele cu cross-validation
- Antrena modelul final (LinearSVC) pe setul de antrenament
- Salva modelul in final_product_classifier.pkl

## La final, vei vedea:

- Rezultatele cross-validation pentru fiecare model
- Raportul de evaluare pe setul de test

## Prezicerea categoriei unui produs
Ruleaza scriptul predict_category.py:
python predict_category.py

- Introdu titlul unui produs cand ti se solicita:

Title (or Enter to quit): Samsung Galaxy S23
Predicted: Mobile Phones
Apasa Enter fara a scrie nimic pentru a iesi din interfata.

## Observatii

Modelul foloseste un pipeline: TfidfVectorizer + LinearSVC.
Inputul pentru prezicere este curatat si standardizat similar cu setul de antrenament.
