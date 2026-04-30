"""
Multilingual Medical Symptom Chatbot
======================================
An intelligent chatbot that detects your input language, translates it to English,
matches your symptoms with medical conditions, applies common-disease priors,
validates/reranks via Gemini, and responds in your original language.

Libraries Used:
- langdetect: Fast language detection (supports 55+ languages)
- googletrans: Google Translate API for free translation
- transformers + torch: BERT embeddings for semantic text matching
- google-generativeai: Gemini LLM for validation and reranking
- pandas: Data manipulation and analysis
- numpy: Numerical operations
"""

import os
import re
import json
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional

# Language detection and translation
try:
    from langdetect import detect, detect_langs
    from googletrans import Translator
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    print("ERROR: Required libraries not installed.")
    print("Install them using: pip install langdetect googletrans transformers torch")
    exit(1)

# Google Gemini
try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: google-generativeai not installed.")
    print("Install it using: pip install google-generativeai")
    exit(1)

# Kaggle Secrets — configure Gemini API key
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GEMINI_API_KEY = user_secrets.get_secret("gemini_key")
    genai.configure(api_key=GEMINI_API_KEY)
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
except Exception as e:
    print(f"Warning: Could not load Gemini API key from Kaggle Secrets: {e}")
    print("Set GEMINI_API_KEY manually if running outside Kaggle.")
    _key = os.environ.get("GEMINI_API_KEY", "")
    if _key:
        genai.configure(api_key=_key)
    else:
        print("Warning: GEMINI_API_KEY not set. GeminiValidator will use fallback mode.")

warnings.filterwarnings('ignore')


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
SIMILARITY_THRESHOLD = 0.3
TOP_N_MATCHES        = 3
BERT_MODEL_NAME      = "google/bert_uncased_L-2_H-128_A-2"
BERT_BATCH_SIZE      = 32
GEMINI_MODEL_NAME    = "gemini-2.5-flash"

# Maximum candidates kept after prior injection before passing to Gemini.
# Using 5 gives Gemini richer context while keeping the list manageable.
MAX_CANDIDATES_AFTER_PRIOR = 5

# Similarity boost applied to existing matches that align with a prior keyword.
PRIOR_BOOST = 0.3

# Default similarity score for diseases injected purely from the prior list.
PRIOR_INJECT_SCORE = 0.6

SUPPORTED_LANGUAGES = {
    'en':    'English',
    'es':    'Spanish',
    'fr':    'French',
    'de':    'German',
    'hi':    'Hindi',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'ja':    'Japanese',
    'ar':    'Arabic',
    'pt':    'Portuguese',
    'ru':    'Russian',
    'it':    'Italian',
}

# ============================================================================
# COMMON DISEASE PRIORS
# Maps individual symptom keywords → ordered list of likely common diseases.
# Keys are matched case-insensitively against the English query string.
# ============================================================================
COMMON_DISEASE_PRIORS: Dict[str, List[str]] = {
    # Upper-respiratory / nasal
    "runny nose":        ["common cold", "allergic rhinitis", "flu"],
    "stuffy nose":       ["common cold", "allergic rhinitis", "sinusitis"],
    "blocked nose":      ["common cold", "allergic rhinitis", "sinusitis"],
    "nasal congestion":  ["common cold", "allergic rhinitis", "sinusitis"],
    "sneezing":          ["common cold", "allergic rhinitis", "flu"],

    # Cough / throat
    "dry cough":         ["flu", "covid-19", "bronchitis"],
    "cough":             ["common cold", "bronchitis", "flu"],
    "sore throat":       ["common cold", "strep throat", "flu"],
    "throat pain":       ["common cold", "strep throat", "tonsillitis"],
    "hoarseness":        ["laryngitis", "common cold", "flu"],

    # Fever / systemic
    "fever":             ["flu", "viral infection", "dengue"],
    "high fever":        ["flu", "typhoid", "dengue"],
    "chills":            ["flu", "viral infection", "malaria"],
    "fatigue":           ["flu", "viral infection", "anemia"],
    "body ache":         ["flu", "viral infection", "dengue"],
    "muscle pain":       ["flu", "viral infection", "dengue"],
    "headache":          ["flu", "migraine", "viral infection"],

    # Gastrointestinal
    "nausea":            ["gastroenteritis", "flu", "food poisoning"],
    "vomiting":          ["gastroenteritis", "food poisoning", "flu"],
    "diarrhea":          ["gastroenteritis", "food poisoning", "irritable bowel syndrome"],
    "stomach pain":      ["gastroenteritis", "irritable bowel syndrome", "gastritis"],
    "stomach ache":      ["gastroenteritis", "irritable bowel syndrome", "gastritis"],
    "indigestion":       ["gastritis", "gerd", "irritable bowel syndrome"],
    "bloating":          ["irritable bowel syndrome", "gastritis", "gerd"],
    "heartburn":         ["gerd", "gastritis", "peptic ulcer"],
    "loose motions":     ["gastroenteritis", "food poisoning", "irritable bowel syndrome"],

    # Respiratory
    "shortness of breath": ["asthma", "bronchitis", "pneumonia"],
    "difficulty breathing": ["asthma", "pneumonia", "bronchitis"],
    "wheezing":            ["asthma", "bronchitis", "allergic reaction"],
    "chest pain":          ["angina", "gerd", "costochondritis"],
    "chest tightness":     ["asthma", "angina", "bronchitis"],

    # Eyes / ears / skin
    "itchy eyes":        ["allergic rhinitis", "conjunctivitis", "dry eye"],
    "red eyes":          ["conjunctivitis", "allergic rhinitis", "dry eye"],
    "ear pain":          ["ear infection", "otitis media", "swimmer's ear"],
    "rash":              ["allergic reaction", "eczema", "viral infection"],
    "itching":           ["allergic reaction", "eczema", "scabies"],
    "skin rash":         ["allergic reaction", "eczema", "chickenpox"],

    # Neurological / psychological
    "dizziness":         ["vertigo", "anemia", "hypotension"],
    "fainting":          ["hypotension", "anemia", "vasovagal syncope"],
    "anxiety":           ["generalized anxiety disorder", "panic disorder", "stress"],
    "insomnia":          ["insomnia", "anxiety", "stress"],

    # Musculoskeletal
    "joint pain":        ["arthritis", "gout", "viral infection"],
    "back pain":         ["muscle strain", "herniated disc", "sciatica"],
    "neck pain":         ["muscle strain", "cervical spondylosis", "tension headache"],
    "knee pain":         ["osteoarthritis", "ligament injury", "gout"],

    # Urinary
    "burning urination": ["urinary tract infection", "cystitis", "kidney stone"],
    "frequent urination":["urinary tract infection", "diabetes", "overactive bladder"],
    "blood in urine":    ["urinary tract infection", "kidney stone", "cystitis"],

    # Other common
    "weight loss":       ["diabetes", "hyperthyroidism", "tuberculosis"],
    "excessive thirst":  ["diabetes", "dehydration", "diabetes insipidus"],
    "night sweats":      ["tuberculosis", "menopause", "lymphoma"],
    "swollen glands":    ["common cold", "flu", "lymphadenitis"],
}


# Flatten all common diseases into a set for fast lookup
COMMON_DISEASE_SET = {
    d.lower()
    for diseases in COMMON_DISEASE_PRIORS.values()
    for d in diseases
}

# ============================================================================
# LANGUAGE DETECTION AND TRANSLATION
# ============================================================================
class LanguageProcessor:
    """Handles language detection and translation."""

    def __init__(self):
        self.translator = Translator()

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except Exception as e:
            print(f"Error detecting language: {e}")
            return 'en'

    def translate_to_english(self, text: str, source_lang: str) -> str:
        if source_lang == 'en':
            return text
        try:
            translation = self.translator.translate(
                text, src_language=source_lang, dest_language='en'
            )
            return translation['text']
        except Exception as e:
            print(f"Error translating text: {e}")
            return text

    def translate_to_language(self, text: str, target_lang: str) -> str:
        if target_lang == 'en':
            return text
        try:
            translation = self.translator.translate(
                text, src_language='en', dest_language=target_lang
            )
            return translation['text']
        except Exception as e:
            print(f"Error translating response: {e}")
            return text


# ============================================================================
# GEMINI VALIDATOR
# ============================================================================
class GeminiValidator:
    """
    Uses Google Gemini to validate and rerank disease candidates.

    The validator receives the (possibly prior-boosted) candidate list from
    apply_common_disease_prior() and returns a clean, medically sound ranking.
    """

    _PROMPT_TEMPLATE = (
        'You are a STRICT medical triage assistant.\n'
        'Patient symptoms: "{query}"\n'
        'Top predicted conditions: {candidates}\n'
        'Tasks:\n'
        '1. Remove any conditions that are medically incorrect for these symptoms\n'
        '2. Rank the remaining conditions from most to least likely\n'
        '3. If ALL conditions are incorrect, suggest a more appropriate common disease\n'
        'Rules:\n'
        '* PRIORITIZE common diseases (flu, viral infection, common cold)\n'
        '* DO NOT allow rare or specific diseases to rank above common ones unless strongly justified\n'
        '* Do not suggest severe diseases unless symptoms strongly indicate them\n'
        '* Be conservative and realistic\n'

        'CRITICAL RULE:'
        '* If a common disease like "flu" or "viral infection" is present and fits symptoms, it MUST be ranked first\n'
        'Return ONLY valid JSON in this format:\n'
        '{{ "valid_indices": [0,1], "new_condition": null }}\n'
        'OR if all are wrong:\n'
        '{{ "valid_indices": [], "new_condition": "common cold" }}'
    )

    def __init__(self, model_name: str = GEMINI_MODEL_NAME):
        try:
            self.model = genai.GenerativeModel(model_name)
            self._available = True
        except Exception as e:
            print(f"Warning: GeminiValidator could not initialise model '{model_name}': {e}")
            self.model = None
            self._available = False

    # ------------------------------------------------------------------
    def validate(self, query: str, matches: List[Dict]) -> List[Dict]:
        if not self._available or not matches:
            print("Gemini fallback used")
            return matches

        candidates_text = self._format_candidates(matches)
        prompt = self._PROMPT_TEMPLATE.format(
            query=query,
            candidates=candidates_text,
        )

        raw_response = self._call_gemini(prompt)
        if raw_response is None:
            print("Gemini fallback used")
            return matches

        parsed = self._parse_json(raw_response)
        if parsed is None:
            print("Gemini fallback used")
            return matches

        updated_matches = self._apply_gemini_result(parsed, matches)
        if updated_matches is None:
            print("Gemini fallback used")
            return matches

        print("Gemini reranking applied")
        return updated_matches

    # ------------------------------------------------------------------
    def _format_candidates(self, matches: List[Dict]) -> str:
        lines = []
        for i, m in enumerate(matches):
            lines.append(
                f"{i}: {m['disease']} ({m['department']}) "
                f"— similarity: {m['similarity']:.2f}"
            )
        return "\n".join(lines)

    def _call_gemini(self, prompt: str) -> Optional[str]:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"    Gemini API call failed: {e}")
            return None

    def _parse_json(self, raw: str) -> Optional[Dict]:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            print(f"    Gemini JSON extraction failed. Raw output:\n{raw[:200]}")
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"    Gemini JSON parse error: {e}. Extracted:\n{match.group()[:200]}")
            return None

    def _apply_gemini_result(
        self,
        parsed: Dict,
        original_matches: List[Dict],
    ) -> Optional[List[Dict]]:
        try:
            valid_indices: List[int] = parsed.get("valid_indices", [])
            new_condition: Optional[str] = parsed.get("new_condition", None)

            if valid_indices:
                updated = [
                    original_matches[i]
                    for i in valid_indices
                    if 0 <= i < len(original_matches)
                ]
                return updated if updated else None

            if new_condition:
                return [{
                    'disease':     new_condition.strip().title(),
                    'department':  'General Medicine',
                    'similarity':  0.0,
                    'description': 'Suggested by Gemini as a better fit for the reported symptoms.',
                }]

            return None

        except Exception as e:
            print(f"    Error applying Gemini result: {e}")
            return None


# ============================================================================
# SYMPTOM MATCHING ENGINE
# ============================================================================
class SymptomMatcher:
    """Handles symptom matching using BERT embeddings and cosine similarity."""

    def __init__(self, dataset_path: str):
        print("Loading dataset...")
        self.df = pd.read_csv(dataset_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BERT model ({BERT_MODEL_NAME}) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.model     = AutoModel.from_pretrained(BERT_MODEL_NAME).to(self.device)
        self.model.eval()

        print("Building BERT embeddings (this may take a moment)...")
        self.embedding_matrix = self._encode_texts(
            self.df['input_text'].fillna('').astype(str).tolist()
        )
        print(f"✓ Dataset loaded: {len(self.df)} records")
        print(f"✓ BERT embedding matrix built: {self.embedding_matrix.shape}")

    def _mean_pool(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask   = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        for start in range(0, len(texts), BERT_BATCH_SIZE):
            batch = texts[start: start + BERT_BATCH_SIZE]
            enc   = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors='pt'
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            emb = self._mean_pool(out, enc['attention_mask'])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy().astype(np.float32))
        return np.vstack(all_embeddings)

    def find_matching_diseases(self, query: str, top_n: int = TOP_N_MATCHES) -> List[Dict]:
        query_embedding = self._encode_texts([query])[0]
        similarities    = np.dot(self.embedding_matrix, query_embedding)
        top_indices     = np.argsort(similarities)[-top_n:][::-1]

        matches: List[Dict] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= SIMILARITY_THRESHOLD:
                matches.append({
                    'disease':     self.df.iloc[idx]['disease'],
                    'department':  self.df.iloc[idx]['department'],
                    'similarity':  score,
                    'description': self.df.iloc[idx]['input_text'][:100] + "...",
                })
        return matches


# ============================================================================
# MEDICAL CHATBOT
# ============================================================================
class MedicalChatbot:
    """
    Main chatbot class.

    Pipeline per query
    ------------------
    1.  Detect language
    2.  Translate to English
    3a. BERT symptom matching          → raw candidates
    3b. Common-disease prior layer     → boosted / injected candidates
    3c. Gemini validation / reranking  → final validated list
    4.  Translate response to user's language
    5.  Return structured result dict
    """

    def __init__(self, dataset_path: str):
        self.language_processor = LanguageProcessor()
        self.symptom_matcher    = SymptomMatcher(dataset_path)
        self.gemini_validator   = GeminiValidator()
        print("\n✓ Chatbot initialized and ready!")

    # ------------------------------------------------------------------
    # NEW: Common-disease prior layer
    # ------------------------------------------------------------------
    def apply_common_disease_prior(
        self,
        query: str,
        matches: List[Dict],
    ) -> List[Dict]:
        """
        Boost or inject common diseases based on symptom keywords found in *query*.

        Algorithm
        ---------
        1. Scan COMMON_DISEASE_PRIORS keys against the lower-cased query.
        2. Collect every disease name whose keyword fired.
        3. For each collected disease:
           a. If it already appears in *matches* → add PRIOR_BOOST to its score.
           b. If it is absent → inject a new entry with PRIOR_INJECT_SCORE.
        4. De-duplicate (keep highest-scoring copy of each disease name).
        5. Sort descending by similarity, return the top MAX_CANDIDATES_AFTER_PRIOR.

        Args:
            query   (str):        Lower-cased English symptom query.
            matches (List[Dict]): BERT match list (may be empty).

        Returns:
            List[Dict]: Prioritised candidate list (3–5 entries).
        """
        print("Applying common disease prior...")

        query_lower = query.lower()

        # Collect all diseases triggered by any matching keyword
        triggered_diseases: List[str] = []
        for keyword, diseases in COMMON_DISEASE_PRIORS.items():
            if keyword in query_lower:
                triggered_diseases.extend(diseases)

        if not triggered_diseases:
            # No keyword matched — return BERT results unchanged
            print("After prior:", [m["disease"] for m in matches])
            return matches

        # Work on a deep copy so we never mutate the original list
        updated: List[Dict] = [dict(m) for m in matches]

        # Build a lookup: normalised disease name → index in *updated*
        def _norm(name: str) -> str:
            return name.strip().lower()

        existing_index: Dict[str, int] = {
            _norm(m["disease"]): i for i, m in enumerate(updated)
        }

        for disease_name in triggered_diseases:
            key = _norm(disease_name)
            if key in existing_index:
                # Boost existing match — cap at 1.0 to keep scores meaningful
                idx = existing_index[key]
                updated[idx]["similarity"] = min(
                    1.0, updated[idx]["similarity"] + PRIOR_BOOST
                )
            else:
                # Inject new candidate
                new_entry: Dict = {
                    "disease":     disease_name.strip().title(),
                    "department":  "General Medicine",
                    "similarity":  PRIOR_INJECT_SCORE,
                    "description": "Added via common disease prior",
                }
                updated.append(new_entry)
                existing_index[key] = len(updated) - 1

        # De-duplicate: if the same disease was injected AND boosted keep max score
        # (can happen if BERT returned it under a slightly different casing)
        seen: Dict[str, int] = {}       # norm_name → index in deduped list
        deduped: List[Dict] = []
        for entry in updated:
            key = _norm(entry["disease"])
            if key in seen:
                # Keep the higher score
                if entry["similarity"] > deduped[seen[key]]["similarity"]:
                    deduped[seen[key]]["similarity"] = entry["similarity"]
            else:
                seen[key] = len(deduped)
                deduped.append(entry)

        # Sort descending by similarity and keep the best MAX_CANDIDATES_AFTER_PRIOR
        deduped.sort(key=lambda x: x["similarity"], reverse=True)
        result = deduped[:MAX_CANDIDATES_AFTER_PRIOR]

        print("After prior:", [m["disease"] for m in result])
        return result

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------
    def process_query(self, user_input: str) -> Dict:
        """
        Process user query end-to-end.

        Args:
            user_input (str): User's query in any supported language.

        Returns:
            Dict with keys: user_language, original_input, english_translation,
                            matches, response, confidence.
        """
        # --- Step 1: Detect language -----------------------------------
        print(f"\n[1/5] Detecting language...")
        detected_lang = self.language_processor.detect_language(user_input)
        print(f"    ✓ Detected language: {SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)}")

        # --- Step 2: Translate to English ------------------------------
        print(f"[2/5] Translating to English...")
        english_text = self.language_processor.translate_to_english(user_input, detected_lang)
        print(f"    ✓ English translation: \"{english_text}\"")

        # --- Step 3: Match + Prior + Gemini ----------------------------
        print(f"[3/5] Searching for matching diseases...")

        # 3a — BERT semantic matching
        matches = self.symptom_matcher.find_matching_diseases(english_text)

        if not matches:
            print(f"    ✗ No suitable BERT matches found")
            # Still run the prior layer — it may inject relevant candidates
            matches = self.apply_common_disease_prior(english_text, matches)

            if not matches:
                response = self._generate_no_match_response()
                # Skip Gemini — nothing to validate
                translated_response = self.language_processor.translate_to_language(
                    response['message'], detected_lang
                )
                return {
                    'user_language':       SUPPORTED_LANGUAGES.get(detected_lang, detected_lang),
                    'original_input':      user_input,
                    'english_translation': english_text,
                    'matches':             [],
                    'response':            translated_response,
                    'confidence':          0.0,
                }
        else:
            print(f"    ✓ BERT found {len(matches)} match(es)")

        # 3b — Common-disease prior boost / injection
        matches = self.apply_common_disease_prior(english_text, matches)

        # 3c — Gemini validation / reranking
        print(f"    → Running Gemini validation...")
        validated = self.gemini_validator.validate(english_text, matches)

        # If Gemini failed → fallback ranking
        if validated == matches:
            print("⚠️ Gemini failed → applying fallback ranking")
            matches = self.rerank_without_gemini(matches)
        else:
            matches = validated

        # Safety net: if Gemini wiped the list entirely, restore prior output
        if not matches:
            print("Gemini fallback used")
            matches = self.apply_common_disease_prior(
                english_text,
                self.symptom_matcher.find_matching_diseases(english_text),
            )

        response = self._generate_response(matches) if matches else self._generate_no_match_response()

        # --- Step 4: Translate response --------------------------------
        print(f"[4/5] Translating response to "
              f"{SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)}...")
        translated_response = self.language_processor.translate_to_language(
            response['message'], detected_lang
        )
        response['message'] = translated_response
        print(f"    ✓ Response translated")

        # --- Step 5: Return result -------------------------------------
        print(f"[5/5] Complete!")

        return {
            'user_language':       SUPPORTED_LANGUAGES.get(detected_lang, detected_lang),
            'original_input':      user_input,
            'english_translation': english_text,
            'matches':             matches,
            'response':            response['message'],
            'confidence':          matches[0]['similarity'] if matches else 0.0,
        }

    # ------------------------------------------------------------------
    def _generate_response(self, matches: List[Dict]) -> Dict:
        top_match = matches[0]
        if len(matches) == 1:
            message = (
                f"Based on your symptoms, the most likely condition is: "
                f"{top_match['disease']}. "
                f"This falls under the {top_match['department']} department. "
                f"Confidence: {top_match['similarity']:.1%}"
            )
        else:
            message = "Based on your symptoms, the most likely conditions are:\n"
            for i, match in enumerate(matches, 1):
                message += (
                    f"{i}. {match['disease']} ({match['department']}) "
                    f"- Confidence: {match['similarity']:.1%}\n"
                )
            message += (
                f"\nThe top recommendation is: {top_match['disease']} "
                f"(Department: {top_match['department']})"
            )
        return {'message': message}

    def _generate_no_match_response(self) -> Dict:
        message = (
            "I couldn't find a close match for your symptoms in the database. "
            "Please consult with a medical professional or try describing your "
            "symptoms in more detail. For urgent concerns, please visit an Emergency Department."
        )
        return {'message': message}

    # ------------------------------------------------------------------
    def chat(self):
        print("\n" + "=" * 70)
        print("MULTILINGUAL MEDICAL SYMPTOM CHATBOT  (Gemini-enhanced)")
        print("=" * 70)
        print("\nWelcome! I can understand and respond in multiple languages.")
        print("Describe your symptoms in any language you prefer.")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("-" * 70)

        while True:
            try:
                user_input = input("\n🩺 You: ").strip()
                if not user_input:
                    print("Please describe your symptoms.")
                    continue
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Thank you for using the Medical Chatbot. Stay healthy!")
                    break

                result = self.process_query(user_input)

                print("\n" + "-" * 70)
                if result['matches']:
                    print(f"\n📋 Top Match: {result['matches'][0]['disease']}")
                    print(f"🏥 Department: {result['matches'][0]['department']}")
                    print(f"🎯 Confidence: {result['confidence']:.1%}")
                print(f"\n💬 Response:\n{result['response']}")
                print("-" * 70)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
                continue

    def rerank_without_gemini(self, matches: List[Dict]) -> List[Dict]:
        def _norm(name):
            return name.strip().lower()
    
        for m in matches:
            name = _norm(m["disease"])
            base = m["similarity"]
    
            if name in COMMON_DISEASE_SET:
                # Strong boost for common diseases
                adjusted = base + 0.4
            else:
                # Stronger penalty for rare
                adjusted = base - 0.15
    
            # ✅ Clamp between 0 and 1
            m["similarity"] = max(0.0, min(1.0, adjusted))
    
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches

# ============================================================================
# MAIN
# ============================================================================
def main():
    import sys
    dataset_path = 'symptom_sentence_dataset_with_department.csv'
    try:
        chatbot = MedicalChatbot(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file '{dataset_path}' not found.")
        print("Make sure you've run 'add_department_column.py' first.")
        sys.exit(1)
    chatbot.chat()


if __name__ == "__main__":
    main()
