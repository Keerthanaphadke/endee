from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Program started...")

jobs = [
    "Machine Learning Engineer with Python and NLP",
    "Frontend Developer with React and CSS",
    "Data Scientist with Deep Learning experience",
    "Backend Developer with Java and Spring Boot",
    "AI Engineer with Computer Vision skills"
]

vectorizer = TfidfVectorizer()
job_vectors = vectorizer.fit_transform(jobs)

query = input("\nEnter job role: ")

query_vector = vectorizer.transform([query])

similarities = cosine_similarity(query_vector, job_vectors)[0]

top_indices = similarities.argsort()[::-1][:3]

print("\nTop Matches:\n")
for i in top_indices:
    print(f"{jobs[i]} (Score: {similarities[i]:.2f})")

print("\nExplanation:")
print("Results are based on semantic similarity using vector representations.")