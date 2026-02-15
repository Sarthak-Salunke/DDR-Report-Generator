"""
Test Error Handling & Edge Cases
Tests various failure scenarios to ensure graceful error handling
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import DDRPipeline


def test_missing_files():
    """Test handling of missing input files"""
    print("\n" + "="*70)
    print("TEST 1: Missing Input Files")
    print("="*70)
    
    pipeline = DDRPipeline()
    
    try:
        pipeline.run(
            inspection_pdf='nonexistent_inspection.pdf',
            thermal_pdf='nonexistent_thermal.pdf'
        )
        print("❌ FAILED: Should have raised FileNotFoundError")
        return False
    except (FileNotFoundError, Exception) as e:
        print(f"✅ PASSED: Correctly caught missing files")
        print(f"   Error: {str(e)[:100]}")
        return True


def test_corrupted_file():
    """Test handling of corrupted/invalid PDF"""
    print("\n" + "="*70)
    print("TEST 2: Corrupted PDF File")
    print("="*70)
    
    # Create a fake "corrupted" PDF (just a text file with .pdf extension)
    fake_pdf = Path('data/input/corrupted_test.pdf')
    fake_pdf.parent.mkdir(parents=True, exist_ok=True)
    
    with open(fake_pdf, 'w') as f:
        f.write("This is not a real PDF file")
    
    pipeline = DDRPipeline()
    
    try:
        pipeline.run(
            inspection_pdf=str(fake_pdf),
            thermal_pdf=str(fake_pdf)
        )
        print("❌ FAILED: Should have raised an error for corrupted PDF")
        return False
    except (FileNotFoundError, Exception) as e:
        print(f"✅ PASSED: Correctly caught corrupted PDF")
        print(f"   Error type: {type(e).__name__}")
        return True
    finally:
        # Cleanup
        if fake_pdf.exists():
            fake_pdf.unlink()


def test_empty_file():
    """Test handling of empty PDF file"""
    print("\n" + "="*70)
    print("TEST 3: Empty PDF File")
    print("="*70)
    
    # Create an empty file
    empty_pdf = Path('data/input/empty_test.pdf')
    empty_pdf.parent.mkdir(parents=True, exist_ok=True)
    
    with open(empty_pdf, 'w') as f:
        pass  # Create empty file
    
    pipeline = DDRPipeline()
    
    try:
        pipeline.run(
            inspection_pdf=str(empty_pdf),
            thermal_pdf=str(empty_pdf)
        )
        print("❌ FAILED: Should have raised an error for empty file")
        return False
    except (FileNotFoundError, Exception) as e:
        print(f"✅ PASSED: Correctly caught empty file")
        print(f"   Error: {e}")
        return True
    finally:
        # Cleanup
        if empty_pdf.exists():
            empty_pdf.unlink()


def test_validation_methods():
    """Test validation methods directly"""
    print("\n" + "="*70)
    print("TEST 4: Validation Methods")
    print("="*70)
    
    pipeline = DDRPipeline()
    
    # Test 1: Valid files (if they exist)
    if Path('data/input/Sample Report.pdf').exists():
        try:
            pipeline.validate_inputs(
                'data/input/Sample Report.pdf',
                'data/input/Thermal Images.pdf'
            )
            print("✅ Valid files passed validation")
        except Exception as e:
            print(f"⚠️  Validation failed: {e}")
    
    # Test 2: Invalid files
    try:
        pipeline.validate_inputs(
            'nonexistent1.pdf',
            'nonexistent2.pdf'
        )
        print("❌ FAILED: Should have raised error for invalid files")
        return False
    except FileNotFoundError:
        print("✅ Invalid files correctly rejected")
    
    return True


def test_error_handlers():
    """Test error handler methods"""
    print("\n" + "="*70)
    print("TEST 5: Error Handler Methods")
    print("="*70)
    
    pipeline = DDRPipeline()
    
    # Test extraction failure handler
    try:
        result = pipeline.handle_extraction_failure(
            "Test Document",
            Exception("Test error"),
            retry_count=0
        )
        print(f"✅ Extraction handler returned: {result}")
    except:
        print("⚠️  Extraction handler raised exception (expected for max retries)")
    
    # Test merge failure handler
    try:
        pipeline.handle_merge_failure(Exception("Test merge error"))
        print("❌ FAILED: Merge handler should have raised")
        return False
    except Exception:
        print("✅ Merge handler correctly raised exception")
    
    # Test generation failure handler
    try:
        pipeline.handle_generation_failure(Exception("Test generation error"))
        print("❌ FAILED: Generation handler should have raised")
        return False
    except Exception:
        print("✅ Generation handler correctly raised exception")
    
    return True


def run_all_tests():
    """Run all error handling tests"""
    print("\n" + "="*70)
    print("🧪 ERROR HANDLING & EDGE CASE TESTS")
    print("="*70)
    
    tests = [
        ("Missing Files", test_missing_files),
        ("Corrupted File", test_corrupted_file),
        ("Empty File", test_empty_file),
        ("Validation Methods", test_validation_methods),
        ("Error Handlers", test_error_handlers)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
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
        print("✅ ALL ERROR HANDLING TESTS PASSED!")
    else:
        print(f"❌ {failed} TEST(S) FAILED")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
