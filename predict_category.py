from joblib import load

# Incarcam direct pipeline-ul
model = load('final_product_classifier.pkl')

def predict_one(title):
    pred = model.predict([title])[0]
    return pred

if __name__ == "__main__":
    while True:
        t = input("Title (or Enter to quit): ")
        if not t:
            break
        print("Predicted:", predict_one(t))