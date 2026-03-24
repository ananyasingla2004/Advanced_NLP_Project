"""
Multilingual Medical Symptom Chatbot (Improved Version)
------------------------------------------------------
Features:
- Multilingual input (no translation needed for matching)
- High accuracy using Sentence Transformers
- Language detection
- Response translated back to user's language
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from googletrans import Translator
import warnings

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
SIMILARITY_THRESHOLD = 0.4
TOP_N_MATCHES = 3

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ar": "Arabic",
    "zh-cn": "Chinese",
    "ja": "Japanese",
}


# =========================
# LANGUAGE PROCESSOR
# =========================
class LanguageProcessor:
    def __init__(self):
        self.translator = Translator()

    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return "en"

    def translate(self, text, dest_lang):
        if dest_lang == "en":
            return text

        try:
            result = self.translator.translate(text, src="en", dest=dest_lang)
            return result.text
        except:
            return text


# =========================
# SYMPTOM MATCHER (EMBEDDINGS)
# =========================
class SymptomMatcher:
    def __init__(self, dataset_path):
        print("📂 Loading dataset...")
        self.df = pd.read_csv(dataset_path)

        print("🤖 Loading multilingual model...")
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        print("⚙️ Encoding dataset...")
        self.embeddings = self.model.encode(
            self.df["input_text"].astype(str).tolist(), show_progress_bar=True
        )

        print(f"✅ Ready! Loaded {len(self.df)} records")

    def find_matches(self, query):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = np.argsort(similarities)[-TOP_N_MATCHES:][::-1]

        matches = []
        for idx in top_indices:
            score = similarities[idx]

            if score >= SIMILARITY_THRESHOLD:
                matches.append(
                    {
                        "disease": self.df.iloc[idx]["disease"],
                        "department": self.df.iloc[idx]["department"],
                        "similarity": float(score),
                    }
                )

        return matches


# =========================
# CHATBOT
# =========================
class MedicalChatbot:
    def __init__(self, dataset_path):
        self.lang_processor = LanguageProcessor()
        self.matcher = SymptomMatcher(dataset_path)
        print("\n🚀 Chatbot ready!\n")

    def generate_response(self, matches):
        if not matches:
            return (
                "I couldn't find a close match for your symptoms. "
                "Please describe them in more detail or consult a doctor."
            )

        top = matches[0]

        response = f"""
Based on your symptoms, here are the most likely conditions:

1. {top['disease']} ({top['department']})
Confidence: {top['similarity']:.1%}
"""

        if len(matches) > 1:
            response += "\nOther possibilities:\n"
            for m in matches[1:]:
                response += (
                    f"- {m['disease']} ({m['department']}) ({m['similarity']:.1%})\n"
                )

        response += "\n⚠️ This is not a medical diagnosis. Please consult a doctor."

        return response

    def process_query(self, user_input):
        print("\n🔍 Detecting language...")
        lang = self.lang_processor.detect_language(user_input)
        print(f"🌐 Language: {SUPPORTED_LANGUAGES.get(lang, lang)}")

        print("🧠 Matching symptoms...")
        matches = self.matcher.find_matches(user_input)

        response = self.generate_response(matches)

        print("🔁 Translating response...")
        translated = self.lang_processor.translate(response, lang)

        return translated

    def chat(self):
        print("=" * 60)
        print("🩺 MULTILINGUAL MEDICAL CHATBOT")
        print("=" * 60)
        print("Type 'exit' to quit\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            try:
                reply = self.process_query(user_input)
                print("\nBot:", reply)
                print("-" * 60)

            except Exception as e:
                print("⚠️ Error:", e)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    DATASET_PATH = "symptom_sentence_dataset_with_department.csv"

    try:
        bot = MedicalChatbot(DATASET_PATH)
        bot.chat()
    except FileNotFoundError:
        print("❌ Dataset not found. Check file path.")
