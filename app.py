from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import mean_squared_error
import numpy as np
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)
# HOME
@app.route('/')
def home():
    return render_template('index.html')
# LINEAR REGRESSION
@app.route('/linear_regression', methods=['POST'])
def linear_regression():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    df = pd.read_csv(filepath)
    df = df.dropna()
    numeric_df = df.select_dtypes(include=['number'])
    columns = numeric_df.columns
    X = numeric_df[[columns[0]]]
    y = numeric_df[columns[1]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    plt.figure()
    plt.scatter(X_test, y_test)
    plt.plot(X_test, predictions, color='red')
    regression_plot = 'static/plots/regression.png'
    plt.savefig(regression_plot)
    plt.close()
    return render_template(
        'result.html',
        mse=mse,
        regression_plot=regression_plot
    )
# KMEANS
@app.route('/kmeans', methods=['POST'])
def kmeans():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    dataset = pd.read_csv(filepath, skipinitialspace=True)
    print(dataset)
    algo = KMeans(n_clusters=2)
    y_predicted = algo.fit_predict(dataset[['Height', 'Weight']])
    dataset['cluster'] = y_predicted
    centers = algo.cluster_centers_
    dataset1 = dataset[dataset.cluster == 0]
    dataset2 = dataset[dataset.cluster == 1]
    # Plot cluster 0
    plt.figure()
    plt.scatter(dataset1['Height'], dataset1['Weight'], color='green', label='Cluster 0')
    # Plot cluster 1
    plt.scatter(dataset2['Height'], dataset2['Weight'], color='red', label='Cluster 1')
    # Plot centroids
    plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='*', label='Centroid')
    # Labels and legend
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.legend()
    kmeans_plot = 'static/plots/kmeans.png'
    plt.savefig(kmeans_plot)
    plt.close()
    return render_template(
        'result.html',
        kmeans_plot=kmeans_plot
    )
# DBSCAN
@app.route('/dbscan', methods=['POST'])
def dbscan():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    df = pd.read_csv(filepath)
    df = df.dropna()
    numeric_df = df.select_dtypes(include=['number'])
    data = numeric_df.values
    epsilon = 5.0
    min_samples = 5
    # Applying DBSCAN
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples)
    clusters = clustering.fit_predict(data)
    unique_clusters = set(clusters)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_clusters))]
    plt.figure()
    for k, col in zip(unique_clusters, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (clusters == k)
        xy = data[class_member_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            'o',
            markerfacecolor=tuple(col),
            markeredgecolor='k',
            markersize=14
        )
    plt.title(f"DBSCAN Clustering (epsilon={epsilon}, min_samples={min_samples})")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid()
    dbscan_plot = 'static/plots/dbscan.png'
    plt.savefig(dbscan_plot)
    plt.close()
    return render_template(
        'result.html',
        dbscan_plot=dbscan_plot
    )


@app.route('/cnn', methods=['POST'])
def cnn():
    from deepface import DeepFace

    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        result = DeepFace.analyze(
            img_path=filepath,
            actions=['gender'],
            enforce_detection=False
        )

        gender_scores = result[0]['gender']
        predicted_class = max(gender_scores, key=gender_scores.get)
        confidence = round(gender_scores[predicted_class], 2)

    except Exception as e:
        return render_template(
            'result.html',
            cnn_result=f"Error: {str(e)}",
            cnn_confidence=0
        )

    return render_template(
        'result.html',
        cnn_result=predicted_class,
        cnn_confidence=confidence
    )
# SENTIMENT ANALYSIS
@app.route('/sentiment', methods=['POST'])
def sentiment():
    from transformers import pipeline
    text_input = request.form.get('text_input', '')
    sentiment_pipeline = pipeline("sentiment-analysis")
    results = sentiment_pipeline([text_input])
    result = results[0]
    return render_template(
        'result.html',
        sentiment_text=text_input,
        sentiment_label=result['label'],
        sentiment_score=round(result['score'], 4)
    )
# QUESTION ANSWERING
@app.route('/qa', methods=['POST'])
def qa():
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    import torch

    question = request.form.get('question', '')
    context = request.form.get('context', '')

    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
    )

    import torch.nn.functional as F
    score = float(F.softmax(outputs.start_logits, dim=1).max())

    return render_template(
        'result.html',
        qa_question=question,
        qa_answer=answer,
        qa_score=round(score, 4)
    )
@app.route('/textgen', methods=['POST'])
def textgen():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    prompt = request.form.get('prompt', '')

    model_name = "facebook/opt-125m"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return render_template(
        'result.html',
        textgen_prompt=prompt,
        textgen_output=generated_text
    )
# TRANSLATION (English to Urdu)
@app.route('/translate', methods=['POST'])
def translate():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    text_input = request.form.get('text_input', '')

    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(text_input, return_tensors="pt", padding=True)

    translated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("urd_Arab"),
        max_length=400
    )

    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return render_template(
        'result.html',
        translation_input=text_input,
        translation_output=translated_text
    )

# NER
@app.route('/ner', methods=['POST'])
def ner():
    from transformers import pipeline

    text_input = request.form.get('text_input', '')

    # Auto capitalize first letter of each word for better NER
    text_capitalized = text_input.title()

    nlp = pipeline("ner", model="dslim/bert-base-NER")
    ner_results = nlp(text_capitalized)

    # Group subword tokens (##) into full words
    entities = []
    current_word = ""
    current_entity = ""
    current_score = 0
    count = 0

    for entity in ner_results:
        word = entity['word']
        if word.startswith("##"):
            current_word += word[2:]
            current_score += entity['score']
            count += 1
        else:
            if current_word:
                entities.append({
                    'word': current_word,
                    'entity': current_entity,
                    'score': round(current_score / count, 4)
                })
            current_word = word
            current_entity = entity['entity']
            current_score = entity['score']
            count = 1

    if current_word:
        entities.append({
            'word': current_word,
            'entity': current_entity,
            'score': round(current_score / count, 4)
        })

    return render_template(
        'result.html',
        ner_input=text_input,
        ner_entities=entities
    )
# APRIORI
@app.route('/apriori', methods=['POST'])
def apriori_route():
    from mlxtend.frequent_patterns import apriori, association_rules

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

   
    transactions = []
    with open(filepath, 'r') as f:
        for line in f:
            items = [item.strip() for item in line.strip().split(',') if item.strip()]
            if items:
                transactions.append(items)

   
    all_items = sorted(set(item for transaction in transactions for item in transaction))


    encoded_data = pd.DataFrame([
        {item: (item in transaction) for item in all_items}
        for transaction in transactions
    ])

    min_support = float(request.form.get('min_support', 0.5))
    min_confidence = float(request.form.get('min_confidence', 0.7))

    frequent_itemsets = apriori(
        encoded_data,
        min_support=min_support,
        use_colnames=True
    )

    if frequent_itemsets.empty:
        return render_template(
            'result.html',
            apriori_itemsets=[],
            apriori_rules=[],
            apriori_message="Koi frequent itemsets nahi mile. Min Support kam karo."
        )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
        num_itemsets=len(frequent_itemsets)
    )

    rules_list = []
    for _, row in rules.iterrows():
        rules_list.append({
            'antecedents': ', '.join(list(row['antecedents'])),
            'consequents': ', '.join(list(row['consequents'])),
            'support': round(row['support'], 4),
            'confidence': round(row['confidence'], 4),
            'lift': round(row['lift'], 4)
        })

    itemsets_list = []
    for _, row in frequent_itemsets.iterrows():
        itemsets_list.append({
            'itemset': ', '.join(list(row['itemsets'])),
            'support': round(row['support'], 4)
        })

    return render_template(
        'result.html',
        apriori_itemsets=itemsets_list,
        apriori_rules=rules_list
    )
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)