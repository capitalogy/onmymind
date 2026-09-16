import os
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load the lightweight SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample song database (Expand or load from JSON later)
SONG_DATABASE = [
    {
        "title": "Blinding Lights",
        "artist": "The Weeknd",
        "lyrics": "I said, ooh, I'm blinded by the lights. No, I can't sleep until I feel your touch."
    },
    {
        "title": "Hotel California",
        "artist": "Eagles",
        "lyrics": "On a dark desert highway, cool wind in my hair. Warm smell of colitas, rising up through the air."
    },
    {
        "title": "On My Mind",
        "artist": "Ellie Goulding",
        "lyrics": "You wanted my heart, but I'm just plain insane. You're on my mind, on my mind."
    }
]

# Pre-encode song lyrics into vector embeddings at startup
corpus_sentences = [song['lyrics'] for song in SONG_DATABASE]
corpus_embeddings = model.encode(corpus_sentences, convert_to_tensor=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    user_query = data.get('query', '')
    
    if not user_query:
        return jsonify({"results": []})

    # Convert user query to vector space
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    
    # Calculate cosine similarity scores
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]
    
    results = []
    for hit in hits:
        song = SONG_DATABASE[hit['corpus_id']]
        results.append({
            "title": song["title"],
            "artist": song["artist"],
            "lyrics": song["lyrics"],
            "score": round(float(hit['score']), 4)
        })

    return jsonify({"results": results})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)