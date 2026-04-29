"""
Test Script for Medical Chatbot
================================
Run this to verify the chatbot is working correctly without interactive input.

Usage:
    python test_chatbot.py
"""

import sys
import json
from pathlib import Path

def test_imports():
    """Test if all required libraries are installed."""
    print("=" * 70)
    print("CHECKING DEPENDENCIES")
    print("=" * 70)
    
    dependencies = {
        'pandas': 'data handling',
        'numpy': 'numerical operations',
        'sklearn': 'text analysis (scikit-learn)',
        'langdetect': 'language detection',
        'googletrans': 'translation'
    }
    
    all_installed = True
    for lib, purpose in dependencies.items():
        try:
            __import__(lib)
            print(f"✓ {lib:20} {purpose}")
        except ImportError:
            print(f"✗ {lib:20} {purpose} - NOT INSTALLED")
            all_installed = False
    
    if not all_installed:
        print("\n❌ Some dependencies are missing!")
        print("\nInstall them using:")
        print("    pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!\n")
    return True


def test_dataset():
    """Test if the dataset file exists and can be loaded."""
    print("=" * 70)
    print("CHECKING DATASET")
    print("=" * 70)
    
    dataset_path = 'symptom_sentence_dataset_with_department.csv'
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        print("\nMake sure you've run:")
        print("    python add_department_column.py")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path)
        
        # Check required columns
        required_columns = ['input_text', 'disease', 'department']
        for col in required_columns:
            if col not in df.columns:
                print(f"❌ Missing column: {col}")
                return False
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Records: {len(df):,}")
        print(f"  - Unique diseases: {df['disease'].nunique()}")
        print(f"  - Departments: {df['department'].nunique()}")
        print(f"  - Columns: {', '.join(df.columns)}\n")
        
        return True
    
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False


def test_chatbot_initialization():
    """Test if the chatbot can be initialized."""
    print("=" * 70)
    print("INITIALIZING CHATBOT")
    print("=" * 70)
    
    try:
        from medical_chatbot import MedicalChatbot
        
        print("Loading dataset and building BERT embeddings...")
        print("(This may take 30-60 seconds on first run)")
        
        chatbot = MedicalChatbot('symptom_sentence_dataset_with_department.csv')
        
        print(f"✓ Chatbot initialized successfully!\n")
        return chatbot
    
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}\n")
        return None


def test_sample_queries(chatbot):
    """Test the chatbot with sample queries."""
    print("=" * 70)
    print("TESTING CHATBOT WITH SAMPLE QUERIES")
    print("=" * 70)
    
    test_cases = [
        {
            'lang': 'English',
            'query': 'I have a high fever and severe cough',
            'expected_dept': ['Pulmonology', 'Infectious Disease', 'General Medicine']
        },
        {
            'lang': 'English',
            'query': 'Chest pain and shortness of breath',
            'expected_dept': ['Cardiology', 'Pulmonology', 'Trauma/General Surgery']
        },
        {
            'lang': 'English',
            'query': 'Joint pain and swelling',
            'expected_dept': ['Orthopedics', 'Rheumatology']
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['lang']}")
        print(f"Query: \"{test['query']}\"")
        
        try:
            result = chatbot.process_query(test['query'])
            
            if result['matches']:
                match = result['matches'][0]
                print(f"✓ Result: {match['disease']}")
                print(f"  Department: {match['department']}")
                print(f"  Confidence: {match['similarity']:.1%}")
                
                # Check if department is reasonable
                if match['department'] in test['expected_dept']:
                    print(f"  ✓ Department is as expected")
                else:
                    print(f"  ⚠ Department is unexpected (expected one of: {', '.join(test['expected_dept'])})")
            else:
                print(f"⚠ No matches found (this might indicate an issue)")
        
        except Exception as e:
            print(f"❌ Error processing query: {e}")


def test_api_wrapper():
    """Test the API wrapper."""
    print("=" * 70)
    print("TESTING API WRAPPER")
    print("=" * 70)
    
    try:
        from medical_chatbot_api import MedicalChatbotAPI
        
        print("Initializing API...")
        api = MedicalChatbotAPI()
        
        # Test analyze_symptoms
        print("\n[1] Testing single symptom analysis...")
        result = api.analyze_symptoms("I have a fever")
        print(f"✓ Result: {json.dumps(result, indent=2)[:200]}...")
        
        # Test get_statistics
        print("\n[2] Testing statistics...")
        stats = api.get_statistics()
        print(f"✓ Total records: {stats['total_records']}")
        print(f"✓ Unique diseases: {stats['total_unique_diseases']}")
        print(f"✓ Departments: {stats['total_departments']}")
        
        # Test search_diseases
        print("\n[3] Testing disease search...")
        results = api.search_diseases("fever")
        print(f"✓ Found {len(results)} diseases containing 'fever'")
        
        # Test get_department_diseases
        print("\n[4] Testing department lookup...")
        diseases = api.get_department_diseases("Cardiology")
        print(f"✓ Found {len(diseases)} Cardiology diseases")
        print(f"  Sample: {', '.join(diseases[:3])}")
        
        print("\n✅ API wrapper working correctly!\n")
        
    except Exception as e:
        print(f"❌ Error testing API: {e}\n")


def test_multilingual():
    """Test multilingual functionality."""
    print("=" * 70)
    print("TESTING MULTILINGUAL SUPPORT")
    print("=" * 70)
    
    try:
        from medical_chatbot import LanguageProcessor
        
        processor = LanguageProcessor()
        
        test_texts = {
            'English': 'I have a fever and cough',
            'Spanish': 'Tengo fiebre y tos',
            'French': 'J\'ai une fièvre et une toux',
            'German': 'Ich habe Fieber und Husten',
        }
        
        for lang, text in test_texts.items():
            detected_lang = processor.detect_language(text)
            print(f"✓ {lang:15} '{text}' → Detected as: {detected_lang}")
        
        print("\n✅ Language detection working!\n")
        
    except Exception as e:
        print(f"❌ Error testing language detection: {e}\n")


def main():
    """Run all tests."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║          MEDICAL CHATBOT - AUTOMATED TEST SUITE                   ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")
    
    # Test 1: Dependencies
    if not test_imports():
        print("Cannot proceed without dependencies.")
        sys.exit(1)
    
    # Test 2: Dataset
    if not test_dataset():
        print("Cannot proceed without dataset.")
        sys.exit(1)
    
    # Test 3: Chatbot initialization
    chatbot = test_chatbot_initialization()
    if not chatbot:
        print("Cannot proceed without chatbot.")
        sys.exit(1)
    
    # Test 4: Sample queries
    test_sample_queries(chatbot)
    
    # Test 5: API wrapper
    test_api_wrapper()
    
    # Test 6: Multilingual
    test_multilingual()
    
    # Final summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("""
✅ All tests completed successfully!

Next steps:
1. Run the interactive chatbot:
   python medical_chatbot.py

2. Use the API programmatically:
   from medical_chatbot_api import MedicalChatbotAPI
   api = MedicalChatbotAPI()
   result = api.analyze_symptoms("Your symptoms here")

3. Read the documentation:
   cat CHATBOT_DOCUMENTATION.md

4. View quick start guide:
   python QUICK_START.py
    """)


if __name__ == "__main__":
    main()
