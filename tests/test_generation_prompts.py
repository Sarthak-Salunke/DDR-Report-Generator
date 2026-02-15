"""
Test generation prompts
Validates that all prompts are properly formatted and functional
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prompts.generation_prompts import *

def test_executive_summary_prompt():
    """Test executive summary prompt generation"""
    print("\n" + "="*70)
    print("🧪 TESTING EXECUTIVE SUMMARY PROMPT")
    print("="*70)
    
    property_details = {
        'address': '123 Main Street, London',
        'property_type': 'Residential - Semi-detached',
        'inspection_date': '2026-02-15'
    }
    
    observations_summary = {
        'total': 12,
        'critical': 1,
        'high': 3,
        'medium': 5,
        'low': 3
    }
    
    major_findings = [
        'Significant dampness in hall area requiring immediate attention',
        'Structural crack in kitchen wall',
        'Roof tiles showing signs of deterioration'
    ]
    
    prompt = build_executive_summary_prompt(
        property_details,
        observations_summary,
        major_findings
    )
    
    print("\n📝 Generated Prompt:")
    print("-" * 70)
    print(prompt[:500] + "...")
    print("-" * 70)
    
    # Validate
    assert '{property_details}' not in prompt, "Property details not substituted"
    assert '12' in prompt, "Total observations not included"
    assert 'Significant dampness' in prompt, "Major findings not included"
    
    print("\n✅ Executive summary prompt generation PASSED")
    return True


def test_recommendations_prompt():
    """Test recommendations prompt generation"""
    print("\n" + "="*70)
    print("🧪 TESTING RECOMMENDATIONS PROMPT")
    print("="*70)
    
    observations_summary = "12 defects identified across multiple areas"
    root_causes = [
        'Poor ventilation leading to condensation',
        'Aging building materials',
        'Inadequate maintenance'
    ]
    
    prompt = build_recommendations_prompt(observations_summary, root_causes)
    
    print("\n📝 Generated Prompt:")
    print("-" * 70)
    print(prompt[:400] + "...")
    print("-" * 70)
    
    # Validate
    assert 'Poor ventilation' in prompt, "Root causes not included"
    assert 'IMMEDIATE ACTIONS' in prompt, "Priority sections present"
    
    print("\n✅ Recommendations prompt generation PASSED")
    return True


def test_conflict_note_prompt():
    """Test conflict note prompt generation"""
    print("\n" + "="*70)
    print("🧪 TESTING CONFLICT NOTE PROMPT")
    print("="*70)
    
    conflict = {
        'location': 'Hall',
        'issue_type': 'Dampness',
        'details': 'Severity mismatch: Medium vs High',
        'sources': ['inspection', 'thermal']
    }
    
    prompt = build_conflict_note_prompt(conflict)
    
    print("\n📝 Generated Prompt:")
    print("-" * 70)
    print(prompt)
    print("-" * 70)
    
    # Validate
    assert 'Hall' in prompt, "Location not included"
    assert 'Dampness' in prompt, "Issue type not included"
    assert 'Medium vs High' in prompt, "Conflict details not included"
    
    print("\n✅ Conflict note prompt generation PASSED")
    return True


def test_area_description_prompt():
    """Test area description prompt generation"""
    print("\n" + "="*70)
    print("🧪 TESTING AREA DESCRIPTION PROMPT")
    print("="*70)
    
    area_name = "Kitchen"
    issues = [
        'Wall crack near window',
        'Minor dampness at skirting',
        'Loose floor tiles'
    ]
    
    prompt = build_area_description_prompt(area_name, issues)
    
    print("\n📝 Generated Prompt:")
    print("-" * 70)
    print(prompt)
    print("-" * 70)
    
    # Validate
    assert 'Kitchen' in prompt, "Area name not included"
    assert 'Wall crack' in prompt, "Issues not included"
    
    print("\n✅ Area description prompt generation PASSED")
    return True


def test_observation_enhancement_prompt():
    """Test observation enhancement prompt generation"""
    print("\n" + "="*70)
    print("🧪 TESTING OBSERVATION ENHANCEMENT PROMPT")
    print("="*70)
    
    observation = {
        'location': 'Bedroom',
        'issue_type': 'Dampness',
        'description': 'Damp patch on wall',
        'severity': 'Medium'
    }
    
    prompt = build_observation_enhancement_prompt(observation)
    
    print("\n📝 Generated Prompt:")
    print("-" * 70)
    print(prompt[:400] + "...")
    print("-" * 70)
    
    # Validate
    assert 'Bedroom' in prompt, "Location not included"
    assert 'Damp patch' in prompt, "Description not included"
    
    print("\n✅ Observation enhancement prompt generation PASSED")
    return True


def test_prompt_template_registry():
    """Test prompt template registry"""
    print("\n" + "="*70)
    print("🧪 TESTING PROMPT TEMPLATE REGISTRY")
    print("="*70)
    
    # Test getting valid template
    template = get_prompt_template('executive_summary')
    assert template == EXECUTIVE_SUMMARY_PROMPT, "Template mismatch"
    print("✅ Retrieved 'executive_summary' template")
    
    # Test all templates
    for name in PROMPT_TEMPLATES.keys():
        template = get_prompt_template(name)
        assert template is not None, f"Template '{name}' is None"
        print(f"✅ Retrieved '{name}' template")
    
    # Test invalid template
    try:
        get_prompt_template('invalid_template')
        print("❌ Should have raised KeyError for invalid template")
        return False
    except KeyError as e:
        print(f"✅ Correctly raised KeyError: {str(e)[:50]}...")
    
    print("\n✅ Prompt template registry PASSED")
    return True


def test_prompt_validation():
    """Test prompt input validation"""
    print("\n" + "="*70)
    print("🧪 TESTING PROMPT VALIDATION")
    print("="*70)
    
    # Test valid inputs
    try:
        validate_prompt_inputs(
            property_details={'address': '123 Main St'},
            observations=[1, 2, 3]
        )
        print("✅ Valid inputs accepted")
    except ValueError:
        print("❌ Valid inputs rejected")
        return False
    
    # Test None input
    try:
        validate_prompt_inputs(property_details=None)
        print("❌ None input should have been rejected")
        return False
    except ValueError as e:
        print(f"✅ None input rejected: {str(e)}")
    
    # Test empty string
    try:
        validate_prompt_inputs(description='')
        print("❌ Empty string should have been rejected")
        return False
    except ValueError as e:
        print(f"✅ Empty string rejected: {str(e)}")
    
    # Test empty list
    try:
        validate_prompt_inputs(observations=[])
        print("❌ Empty list should have been rejected")
        return False
    except ValueError as e:
        print(f"✅ Empty list rejected: {str(e)}")
    
    print("\n✅ Prompt validation PASSED")
    return True


def test_all_helper_functions():
    """Test all helper functions"""
    print("\n" + "="*70)
    print("🧪 TESTING ALL HELPER FUNCTIONS")
    print("="*70)
    
    tests = {
        'Root Cause Analysis': lambda: build_root_cause_analysis_prompt([
            {'location': 'Hall', 'description': 'Dampness'},
            {'location': 'Kitchen', 'description': 'Crack'}
        ]),
        'Priority Assessment': lambda: build_priority_assessment_prompt([
            {'location': 'Hall', 'description': 'Dampness', 'severity': 'High'}
        ]),
        'Technical Summary': lambda: build_technical_summary_prompt(
            'Residential',
            '2026-02-15',
            10,
            ['Visual', 'Thermal'],
            {'Dampness': 5, 'Cracks': 3}
        ),
        'Client Explanation': lambda: build_client_explanation_prompt(
            'Interstitial condensation in cavity wall'
        )
    }
    
    for test_name, test_func in tests.items():
        try:
            result = test_func()
            assert result is not None, f"{test_name} returned None"
            assert len(result) > 0, f"{test_name} returned empty string"
            print(f"✅ {test_name} helper function works")
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            return False
    
    print("\n✅ All helper functions PASSED")
    return True


def run_all_tests():
    """Run all prompt tests"""
    print("\n" + "="*70)
    print("RUNNING GENERATION PROMPTS TESTS")
    print("="*70)
    
    tests = [
        ("Executive Summary Prompt", test_executive_summary_prompt),
        ("Recommendations Prompt", test_recommendations_prompt),
        ("Conflict Note Prompt", test_conflict_note_prompt),
        ("Area Description Prompt", test_area_description_prompt),
        ("Observation Enhancement Prompt", test_observation_enhancement_prompt),
        ("Prompt Template Registry", test_prompt_template_registry),
        ("Prompt Validation", test_prompt_validation),
        ("All Helper Functions", test_all_helper_functions)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
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
        print("✅ ALL GENERATION PROMPTS TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
