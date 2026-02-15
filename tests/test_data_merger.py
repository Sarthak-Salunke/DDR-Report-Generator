"""
Comprehensive tests for DataMerger
Tests duplicate detection, merging logic, and conflict handling
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_merger import DataMerger
from src.data_models import Observation

def test_basic_duplicate_detection():
    """Test that obvious duplicates are detected"""
    print("\n" + "="*60)
    print("TEST 1: Basic Duplicate Detection")
    print("="*60)
    
    # Create two observations about same issue
    obs1 = Observation(
        location="Hall",
        issue_type="Dampness",
        description="Dampness observed at skirting level in the hall area",
        severity="Medium",
        source_document="inspection",
        confidence=0.9,
        evidence="Photo 1-7"
    )
    
    obs2 = Observation(
        location="Hall",
        issue_type="Dampness",
        description="Moisture detected at floor skirting in hall",
        severity="Medium",
        source_document="thermal",
        confidence=0.85,
        evidence="RB02380X.JPG"
    )
    
    merger = DataMerger(similarity_threshold=0.85, verbose=True)
    result = merger.merge_observations([obs1], [obs2])
    
    # Should merge to 1 observation
    assert result['total_observations'] == 1, f"Expected 1, got {result['total_observations']}"
    assert result['duplicate_groups'] >= 1, "Should detect duplicate group"
    
    merged_obs = result['merged_observations'][0]
    assert len(merged_obs.sources) == 2, "Should have both sources"
    assert 'inspection' in merged_obs.sources
    assert 'thermal' in merged_obs.sources
    
    print("[PASSED] Duplicates correctly merged")
    return True

def test_different_locations_not_merged():
    """Test that issues in different locations are NOT merged"""
    print("\n" + "="*60)
    print("TEST 2: Different Locations Not Merged")
    print("="*60)
    
    obs1 = Observation(
        location="Hall",
        issue_type="Dampness",
        description="Dampness at skirting level",
        source_document="inspection",
        confidence=0.9
    )
    
    obs2 = Observation(
        location="Kitchen",
        issue_type="Dampness",
        description="Dampness at skirting level",
        source_document="inspection",
        confidence=0.9
    )
    
    merger = DataMerger(similarity_threshold=0.85, verbose=True)
    result = merger.merge_observations([obs1, obs2], [])
    
    # Should keep separate
    assert result['total_observations'] == 2, f"Expected 2, got {result['total_observations']}"
    
    print("[PASSED] Different locations kept separate")
    return True

def test_conflict_detection():
    """Test that severity conflicts are detected"""
    print("\n" + "="*60)
    print("TEST 3: Conflict Detection")
    print("="*60)
    
    obs1 = Observation(
        location="Master Bedroom",
        issue_type="Crack",
        description="Crack on wall surface",
        severity="Low",
        source_document="inspection",
        confidence=0.8
    )
    
    obs2 = Observation(
        location="Master Bedroom",
        issue_type="Crack",
        description="Wall crack detected",
        severity="High",
        source_document="thermal",
        confidence=0.85
    )
    
    merger = DataMerger(similarity_threshold=0.85, verbose=True)
    result = merger.merge_observations([obs1], [obs2])
    
    # Should detect conflict
    conflicts = result['conflicts_detected']
    assert len(conflicts) > 0, "Should detect severity conflict"
    
    print(f"[PASSED] Detected {len(conflicts)} conflict(s)")
    return True

def test_many_observations():
    """Test with realistic number of observations"""
    print("\n" + "="*60)
    print("TEST 4: Multiple Observations")
    print("="*60)
    
    inspection_obs = [
        Observation(
            location="Hall",
            issue_type="Dampness",
            description="Dampness at skirting level",
            source_document="inspection",
            confidence=0.9
        ),
        Observation(
            location="Bedroom",
            issue_type="Dampness",
            description="Skirting dampness in bedroom",
            source_document="inspection",
            confidence=0.88
        ),
        Observation(
            location="Kitchen",
            issue_type="Crack",
            description="Small wall crack",
            source_document="inspection",
            confidence=0.85
        ),
        Observation(
            location="Master Bedroom",
            issue_type="Seepage",
            description="Water seepage on wall",
            source_document="inspection",
            confidence=0.92
        ),
    ]
    
    thermal_obs = [
        Observation(
            location="Hall",
            issue_type="Dampness",
            description="Dampness observed at skirting level in the hall area",
            source_document="thermal",
            confidence=0.87,
            evidence="Temp delta: 5.4°C"
        ),
        Observation(
            location="Kitchen",
            issue_type="Crack",
            description="Small wall crack observed on kitchen wall",
            source_document="thermal",
            confidence=0.80,
            evidence="Temp delta: 4.8°C"
        ),
    ]
    
    merger = DataMerger(similarity_threshold=0.85, verbose=True)
    result = merger.merge_observations(inspection_obs, thermal_obs)
    
    print(f"\nResults:")
    print(f"  Original count: {len(inspection_obs) + len(thermal_obs)}")
    print(f"  Merged count: {result['total_observations']}")
    print(f"  Duplicate groups: {result['duplicate_groups']}")
    print(f"  Deduplication rate: {result['metadata']['deduplication_rate']:.1%}")
    
    # Should have fewer observations after deduplication
    assert result['total_observations'] < len(inspection_obs) + len(thermal_obs)
    
    print("[PASSED] Multiple observations handled correctly")
    return True

def test_threshold_sensitivity():
    """Test different similarity thresholds"""
    print("\n" + "="*60)
    print("TEST 5: Threshold Sensitivity")
    print("="*60)
    
    obs1 = Observation(
        location="Hall",
        issue_type="Dampness",
        description="Dampness observed",
        source_document="inspection",
        confidence=0.9
    )
    
    obs2 = Observation(
        location="Hall",
        issue_type="Dampness",
        description="Moisture problem",
        source_document="thermal",
        confidence=0.85
    )
    
    # Test with strict threshold
    merger_strict = DataMerger(similarity_threshold=0.95, verbose=False)
    result_strict = merger_strict.merge_observations([obs1], [obs2])
    
    # Test with loose threshold
    merger_loose = DataMerger(similarity_threshold=0.70, verbose=False)
    result_loose = merger_loose.merge_observations([obs1], [obs2])
    
    print(f"  Strict (0.95): {result_strict['total_observations']} observations")
    print(f"  Loose (0.70): {result_loose['total_observations']} observations")
    
    # Loose threshold should merge more
    assert result_loose['total_observations'] <= result_strict['total_observations']
    
    print("[PASSED] Threshold affects merging as expected")
    return True

def run_all_tests():
    """Run all merger tests"""
    print("\n" + "="*70)
    print("RUNNING DATA MERGER TEST SUITE")
    print("="*70)
    
    tests = [
        test_basic_duplicate_detection,
        test_different_locations_not_merged,
        test_conflict_detection,
        test_many_observations,
        test_threshold_sensitivity
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"[FAILED] {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("[OK] ALL TESTS PASSED!")
        return True
    else:
        print("[FAILED] SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
