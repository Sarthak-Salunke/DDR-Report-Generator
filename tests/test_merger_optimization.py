"""
Test threshold sensitivity analysis and result saving
Demonstrates optimization features of the Data Merger
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_merger import DataMerger
from src.data_models import Observation

def test_threshold_analysis():
    """Test threshold sensitivity analysis"""
    print("\n" + "="*70)
    print("🧪 TESTING THRESHOLD SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Create sample observations with varying similarity
    inspection_obs = [
        Observation(
            location="Hall",
            issue_type="Dampness",
            description="Dampness observed at skirting level in the hall area",
            severity="Medium",
            source_document="inspection",
            confidence=0.9,
            evidence="Photo 1-7"
        ),
        Observation(
            location="Bedroom",
            issue_type="Dampness",
            description="Skirting dampness in bedroom",
            severity="Medium",
            source_document="inspection",
            confidence=0.88
        ),
        Observation(
            location="Kitchen",
            issue_type="Crack",
            description="Small wall crack",
            severity="Low",
            source_document="inspection",
            confidence=0.85
        ),
    ]
    
    thermal_obs = [
        Observation(
            location="Hall",
            issue_type="Dampness",
            description="Moisture detected at floor skirting in hall",
            severity="Medium",
            source_document="thermal",
            confidence=0.87,
            evidence="Temp delta: 5.4°C"
        ),
        Observation(
            location="Kitchen",
            issue_type="Crack",
            description="Small wall crack observed on kitchen wall",
            severity="Low",
            source_document="thermal",
            confidence=0.80,
            evidence="Temp delta: 4.8°C"
        ),
    ]
    
    # Create merger
    merger = DataMerger(similarity_threshold=0.85, verbose=False)
    
    # Run threshold analysis
    analysis_results = merger.analyze_threshold_sensitivity(
        inspection_obs,
        thermal_obs,
        thresholds=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )
    
    print("\n✅ Threshold analysis complete!")
    return analysis_results

def test_save_results():
    """Test saving merge results"""
    print("\n" + "="*70)
    print("🧪 TESTING RESULT SAVING")
    print("="*70)
    
    # Create sample observations
    obs1 = Observation(
        location="Living Room",
        issue_type="Dampness",
        description="Dampness at skirting level",
        severity="Medium",
        source_document="inspection",
        confidence=0.9,
        evidence="Photo 45"
    )
    
    obs2 = Observation(
        location="Living Room",
        issue_type="Dampness",
        description="Moisture detected at floor level",
        severity="Medium",
        source_document="thermal",
        confidence=0.85,
        evidence="TH_LR_01.jpg"
    )
    
    # Create merger
    merger = DataMerger(similarity_threshold=0.85, verbose=True)
    
    # Merge observations
    result = merger.merge_observations([obs1], [obs2])
    
    # Save results
    output_path = "data/intermediate/test_merge_results.json"
    merger.save_merge_results(result, output_path)
    
    # Verify file was created
    if os.path.exists(output_path):
        print(f"\n✅ Results successfully saved to {output_path}")
        
        # Show file size
        file_size = os.path.getsize(output_path)
        print(f"   File size: {file_size} bytes")
        
        # Read and display summary
        import json
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        print(f"\n📊 Saved data summary:")
        print(f"   - Total observations: {saved_data['total_observations']}")
        print(f"   - Duplicate groups: {saved_data['duplicate_groups']}")
        print(f"   - Conflicts: {len(saved_data['conflicts'])}")
        print(f"   - Deduplication rate: {saved_data['metadata']['deduplication_rate']:.1%}")
        
        return True
    else:
        print(f"\n❌ Failed to save results to {output_path}")
        return False

def test_config_threshold():
    """Test reading threshold from config"""
    print("\n" + "="*70)
    print("🧪 TESTING CONFIG-BASED THRESHOLD")
    print("="*70)
    
    import yaml
    
    # Read config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    threshold = config.get('deduplication', {}).get('similarity_threshold', 0.85)
    
    print(f"\n📋 Current threshold in config.yaml: {threshold}")
    
    # Create merger with config threshold
    merger = DataMerger(similarity_threshold=threshold, verbose=True)
    
    print(f"✅ Merger initialized with threshold: {merger.similarity_threshold}")
    
    print("\n💡 To adjust threshold:")
    print("   1. Edit config.yaml")
    print("   2. Update deduplication.similarity_threshold")
    print("   3. Recommended values:")
    print("      - 0.70-0.80: More duplicates detected (aggressive)")
    print("      - 0.85: Balanced (recommended)")
    print("      - 0.90-0.95: Fewer duplicates (conservative)")
    
    return True

def run_all_tests():
    """Run all optimization tests"""
    print("\n" + "="*70)
    print("RUNNING DATA MERGER OPTIMIZATION TESTS")
    print("="*70)
    
    tests = [
        ("Threshold Analysis", test_threshold_analysis),
        ("Save Results", test_save_results),
        ("Config Threshold", test_config_threshold)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*70}")
            print(f"Running: {test_name}")
            print(f"{'='*70}")
            
            result = test_func()
            if result or result is None:
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("✅ ALL OPTIMIZATION TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
