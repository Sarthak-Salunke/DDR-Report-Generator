"""
Integration test: Merger with real PDF extraction
Tests the complete pipeline: Parse → Extract → Merge
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.document_parser import DocumentParser
from src.data_extractor import DataExtractor
from src.data_merger import DataMerger
from src.data_models import Observation
from dotenv import load_dotenv

def test_full_extraction_and_merge():
    """Test complete pipeline: Parse → Extract → Merge"""
    print("\n" + "="*70)
    print("🔬 INTEGRATION TEST: Full Pipeline")
    print("="*70)
    
    # Load environment
    load_dotenv()
    
    # Check for LLM API keys (Gemini or Groq)
    gemini_key = os.getenv('GOOGLE_API_KEY')
    groq_key = os.getenv('GROQ_API_KEY')
    
    if not gemini_key or 'your_' in gemini_key:
        if not groq_key or 'your_' in groq_key:
            print("⚠️  Skipping - No LLM API keys configured")
            print("   Set GOOGLE_API_KEY or GROQ_API_KEY in .env")
            print("   See API_KEY_SETUP_GUIDE.md for instructions")
            return True  # Skip, don't fail
    
    try:
        # Step 1: Parse documents
        print("\n📄 Step 1: Parsing documents...")
        parser = DocumentParser(verbose=True)
        
        # Check if sample files exist
        inspection_path = 'data/input/Sample Report.pdf'
        thermal_path = 'data/input/Thermal Images.pdf'
        
        if not os.path.exists(inspection_path):
            print(f"⚠️  Sample file not found: {inspection_path}")
            print("   Skipping integration test")
            return True
        
        if not os.path.exists(thermal_path):
            print(f"⚠️  Sample file not found: {thermal_path}")
            print("   Skipping integration test")
            return True
        
        inspection_doc = parser.extract_text_from_pdf(inspection_path)
        thermal_doc = parser.extract_text_from_pdf(thermal_path)
        
        print(f"  ✓ Inspection report: {len(inspection_doc.text)} characters")
        print(f"  ✓ Thermal report: {len(thermal_doc.text)} characters")
        
        # Step 2: Extract data
        print("\n🧠 Step 2: Extracting with LLM (Multi-Provider System)...")
        extractor = DataExtractor(config_path="config.yaml", verbose=True)
        
        print("\n  Extracting inspection data...")
        inspection_data = extractor.extract_inspection_data(inspection_doc.text)
        
        print(f"\n  Extracting thermal data...")
        thermal_data = extractor.extract_thermal_data(thermal_doc.text)
        
        print(f"\n  ✓ Extracted {len(inspection_data.observations)} inspection observations")
        print(f"  ✓ Extracted {len(thermal_data.readings)} thermal readings")
        
        # Step 3: Convert thermal readings to observations
        print("\n🔄 Step 3: Converting thermal readings to observations...")
        thermal_observations = []
        for reading in thermal_data.readings:
            # Use location from reading, default to "Unknown" if not available
            location = reading.location if hasattr(reading, 'location') and reading.location and reading.location != "Not Available" else "Unknown"
            
            obs = Observation(
                location=location,
                issue_type="Thermal Anomaly",
                description=reading.interpretation,
                source_document="thermal",
                confidence=0.8,
                evidence=f"Hotspot: {reading.hotspot_temp}°C, Delta: {reading.temperature_delta}°C"
            )
            thermal_observations.append(obs)
        
        print(f"  ✓ Converted {len(thermal_observations)} thermal observations")
        
        # Step 4: Merge
        print("\n🔗 Step 4: Merging observations...")
        merger = DataMerger(similarity_threshold=0.85, verbose=True)
        result = merger.merge_observations(
            inspection_data.observations,
            thermal_observations
        )
        
        # Verify results
        print("\n" + "="*70)
        print("MERGE RESULTS:")
        print("="*70)
        original_count = len(inspection_data.observations) + len(thermal_observations)
        print(f"  Original observations: {original_count}")
        print(f"  Merged observations: {result['total_observations']}")
        print(f"  Duplicate groups: {result['duplicate_groups']}")
        print(f"  Conflicts detected: {len(result['conflicts_detected'])}")
        print(f"  Deduplication rate: {result['metadata']['deduplication_rate']:.1%}")
        
        # Show sample merged observations
        if result['merged_observations']:
            print("\n📋 Sample merged observations:")
            for i, sample in enumerate(result['merged_observations'][:3], 1):
                print(f"\n  {i}. Location: {sample.location}")
                print(f"     Issue: {sample.issue_type}")
                print(f"     Sources: {sample.sources}")
                print(f"     Evidence count: {len(sample.evidence)}")
                print(f"     Confidence: {sample.confidence:.2f}")
                if sample.is_duplicate:
                    print(f"     ⚠️  Merged from {len(sample.sources)} sources")
                if sample.conflict_detected:
                    print(f"     ⚠️  Conflict: {sample.conflict_details}")
        
        # Show conflicts if any
        if result['conflicts_detected']:
            print("\n⚠️  Conflicts requiring review:")
            for i, conflict in enumerate(result['conflicts_detected'], 1):
                print(f"\n  {i}. {conflict['location']} - {conflict['issue_type']}")
                print(f"     Type: {conflict['conflict_type']}")
                print(f"     Details: {conflict['details']}")
                print(f"     Action: {conflict['action_required']}")
        
        # Validation checks
        print("\n" + "="*70)
        print("VALIDATION CHECKS:")
        print("="*70)
        
        checks = {
            "Observations extracted": len(inspection_data.observations) > 0,
            "Thermal readings extracted": len(thermal_data.readings) > 0,
            "Merge reduced duplicates": result['total_observations'] <= original_count,
            "All observations have locations": all(obs.location for obs in result['merged_observations']),
            "All observations have descriptions": all(obs.description for obs in result['merged_observations']),
            "Confidence scores valid": all(0 <= obs.confidence <= 1 for obs in result['merged_observations'])
        }
        
        all_passed = True
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n✅ ALL INTEGRATION TESTS PASSED!")
            return True
        else:
            print("\n⚠️  Some validation checks failed")
            return False
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_extraction_and_merge()
    sys.exit(0 if success else 1)
