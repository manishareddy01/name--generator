from flask import Flask, render_template, request, session
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Function to extract syllables from words
def extract_syllables(words):
    syllables = []
    for word in words:
        word_length = len(word)
        if word_length >= 3:
            syllables.append(word[:2])
            syllables.append(word[-2:])
        else:
            syllables.append(word)
    return syllables

# Function to find closest name in dataset for a given gender and first letter
def find_closest_name(input_words, dataset, gender, first_letter, previous_names=None):
    if previous_names is None:
        previous_names = set()
    filtered_data = [entry['name'] for entry in dataset if entry['gender'] == gender and entry['name'].startswith(first_letter)]
    if not filtered_data:
        return None
    vectorizer = TfidfVectorizer(analyzer='char')
    tfidf_matrix = vectorizer.fit_transform(filtered_data)
    input_words_tfidf = vectorizer.transform(input_words)
    cosine_similarities = cosine_similarity(input_words_tfidf, tfidf_matrix)
    closest_indices = cosine_similarities.argsort(axis=1)[:, -1]
    closest_name = filtered_data[closest_indices[0]]
    
    # Check if the closest name is in the input or previous names
    while closest_name in input_words or closest_name in previous_names:
        other_names = [name for name in filtered_data if name != closest_name]
        closest_name = random.choice(other_names)
    
    return closest_name

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'regenerate' in request.form:
            input_words = session['input_words']
        else: 
            input_words = request.form['input_words'].split()
        input_words = [word.capitalize() for word in input_words]
        first_letter = input_words[0][0]

        # Load data from JSON file 
        with open('name.json', 'r') as file:
            data = json.load(file)

        if 'regenerate' not in request.form:
            session['input_words'] = input_words
            closest_male_name = find_closest_name(input_words, data, 'male', first_letter)
            closest_female_name = find_closest_name(input_words, data, 'female', first_letter)
            session['previous_names'] = [closest_male_name, closest_female_name]
            session.modified = True
        elif 'regenerate' in request.form: 
            input_words = session['input_words']
            previous_names = session.get('previous_names', [])
            closest_male_name = find_closest_name(input_words, data, 'male', first_letter, set(previous_names))
            closest_female_name = find_closest_name(input_words, data, 'female', first_letter, set(previous_names))
            session['previous_names'].extend([closest_male_name, closest_female_name])
            session.modified = True
    
           

        return render_template('index.html', closest_male_name=closest_male_name, closest_female_name=closest_female_name, regenerate_button=True)

    return render_template('index.html', regenerate_button=False)

if __name__ == "__main__":
    app.run(debug=True)
