"""
End-to-end pipeline testing
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import DDRPipeline


def test_standard_pipeline():
    """Test pipeline with standard input files"""
    print("\n" + "="*70)
    print("TEST 1: Standard Pipeline Execution")
    print("="*70)
    
    # Check if input files exist
    inspection_pdf = 'data/input/Sample Report.pdf'
    thermal_pdf = 'data/input/Thermal Images.pdf'
    
    if not Path(inspection_pdf).exists():
        print(f"⚠️  SKIPPED: Input file not found: {inspection_pdf}")
        return True
    
    if not Path(thermal_pdf).exists():
        print(f"⚠️  SKIPPED: Input file not found: {thermal_pdf}")
        return True
    
    pipeline = DDRPipeline()
    success = pipeline.run()
    
    assert success, "Pipeline should complete successfully"
    
    # Verify output file exists
    output_file = Path('data/output/DDR_Report_Final.docx')
    assert output_file.exists(), "Output DOCX should be created"
    assert output_file.stat().st_size > 10000, "Output file should be substantial (>10KB)"
    
    print("✅ TEST PASSED: Standard pipeline works!")
    return True


def test_custom_output_path():
    """Test pipeline with custom output path"""
    print("\n" + "="*70)
    print("TEST 2: Custom Output Path")
    print("="*70)
    
    # Check if input files exist
    inspection_pdf = 'data/input/Sample Report.pdf'
    thermal_pdf = 'data/input/Thermal Images.pdf'
    
    if not Path(inspection_pdf).exists() or not Path(thermal_pdf).exists():
        print("⚠️  SKIPPED: Input files not found")
        return True
    
    custom_output = 'data/output/DDR_Custom_Test.docx'
    
    pipeline = DDRPipeline()
    success = pipeline.run(output_path=custom_output)
    
    assert success, "Pipeline should complete successfully"
    assert Path(custom_output).exists(), "Custom output should exist"
    
    print(f"✅ TEST PASSED: Custom output created at {custom_output}")
    return True


def test_intermediate_files():
    """Test that intermediate files are saved"""
    print("\n" + "="*70)
    print("TEST 3: Intermediate Files Saved")
    print("="*70)
    
    # Check if input files exist
    inspection_pdf = 'data/input/Sample Report.pdf'
    thermal_pdf = 'data/input/Thermal Images.pdf'
    
    if not Path(inspection_pdf).exists() or not Path(thermal_pdf).exists():
        print("⚠️  SKIPPED: Input files not found")
        return True
    
    pipeline = DDRPipeline()
    pipeline.run()
    
    # Check intermediate files
    intermediate_files = [
        'data/intermediate/parsed_inspection.txt',
        'data/intermediate/parsed_thermal.txt',
        'data/intermediate/extracted_inspection.json',
        'data/intermediate/extracted_thermal.json',
        'data/intermediate/merged_observations.json'
    ]
    
    for filepath in intermediate_files:
        assert Path(filepath).exists(), f"Intermediate file should exist: {filepath}"
        print(f"  ✓ Found: {filepath}")
    
    print("✅ TEST PASSED: All intermediate files saved!")
    return True


def test_error_handling():
    """Test pipeline error handling with missing files"""
    print("\n" + "="*70)
    print("TEST 4: Error Handling (Missing Files)")
    print("="*70)
    
    pipeline = DDRPipeline()
    
    # Try with non-existent files
    success = pipeline.run(
        inspection_pdf='data/input/nonexistent.pdf',
        thermal_pdf='data/input/also_nonexistent.pdf'
    )
    
    assert not success, "Pipeline should fail gracefully with missing files"
    
    print("✅ TEST PASSED: Error handling works correctly!")
    return True


def test_api_key_validation():
    """Test that API key validation works"""
    print("\n" + "="*70)
    print("TEST 5: API Key Validation")
    print("="*70)
    
    # Check if at least one API key is set
    has_api_key = (
        os.getenv('GOOGLE_API_KEY') or
        os.getenv('GROQ_API_KEY') or
        os.getenv('ANTHROPIC_API_KEY')
    )
    
    if has_api_key:
        print("  ✓ API key found")
        try:
            pipeline = DDRPipeline()
            print("  ✓ Pipeline initialized successfully")
        except ValueError as e:
            print(f"  ✗ Pipeline initialization failed: {e}")
            return False
    else:
        print("  ⚠️  No API key found (expected in production)")
        # This should raise an error
        try:
            pipeline = DDRPipeline()
            print("  ✗ Pipeline should have raised error without API key")
            return False
        except ValueError:
            print("  ✓ Correctly raised error for missing API key")
    
    print("✅ TEST PASSED: API key validation works!")
    return True


def run_all_tests():
    """Run all pipeline tests"""
    print("\n" + "="*70)
    print("🧪 RUNNING PIPELINE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Standard Pipeline", test_standard_pipeline),
        ("Custom Output Path", test_custom_output_path),
        ("Intermediate Files", test_intermediate_files),
        ("Error Handling", test_error_handling),
        ("API Key Validation", test_api_key_validation)
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"📊 TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} TEST(S) FAILED")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
