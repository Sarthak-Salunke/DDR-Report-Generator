#!/usr/bin/env python3
"""
DDR Report Generator - Main Pipeline
Orchestrates the complete process: Parse → Extract → Merge → Generate
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import yaml
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_parser import DocumentParser
from src.data_extractor import DataExtractor
from src.data_merger import DataMerger
from src.ddr_generator import DDRGenerator
from src.llm_manager import LLMManager
from src.data_models import Observation
from src.utils import log_step, ensure_directory_exists


class DDRPipeline:
    """
    Main pipeline orchestrator for DDR Report Generation
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize pipeline with configuration"""
        self.start_time = datetime.now()
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Check for API keys
        self.api_key = (
            os.getenv('GOOGLE_API_KEY') or 
            os.getenv('GROQ_API_KEY') or 
            os.getenv('ANTHROPIC_API_KEY')
        )
        
        if not self.api_key:
            raise ValueError(
                "No API key found! Set GOOGLE_API_KEY, GROQ_API_KEY, or ANTHROPIC_API_KEY in .env"
            )
        
        # Initialize components
        self.parser = DocumentParser(verbose=True)
        
        # Initialize extractor (creates its own LLM manager)
        self.extractor = DataExtractor(config_path=config_path, verbose=True)
        
        # Use the extractor's LLM manager for the generator
        self.llm_manager = self.extractor.llm_manager
        
        # Initialize merger
        self.merger = DataMerger(
            similarity_threshold=self.config.get('deduplication', {}).get('similarity_threshold', 0.85),
            verbose=True
        )
        
        # Initialize generator with LLM manager
        self.generator = DDRGenerator(
            llm_manager=self.llm_manager,
            verbose=True
        )
        
        # Create output directories
        ensure_directory_exists('data/output')
        ensure_directory_exists('data/intermediate')
        
        print("\n" + "="*70)
        print("DDR REPORT GENERATOR - INITIALIZED")
        print("="*70)
        print(f"API Provider: {self._detect_provider()}")
        print(f"Primary LLM: {self.config.get('llm', {}).get('primary_provider', 'Unknown')}")
        print(f"Similarity Threshold: {self.config.get('deduplication', {}).get('similarity_threshold', 0.85)}")
        print("="*70 + "\n")
    
    def _detect_provider(self) -> str:
        """Detect which LLM provider is being used"""
        if os.getenv('GOOGLE_API_KEY'):
            return "Google Gemini"
        elif os.getenv('GROQ_API_KEY'):
            return "Groq"
        elif os.getenv('ANTHROPIC_API_KEY'):
            return "Anthropic Claude"
        return "Unknown"
    
    def validate_inputs(self, inspection_pdf: str, thermal_pdf: str):
        """
        Validate input files before processing
        
        Args:
            inspection_pdf: Path to inspection PDF
            thermal_pdf: Path to thermal PDF
            
        Raises:
            FileNotFoundError: If required files are missing
        """
        errors = []
        
        if not Path(inspection_pdf).exists():
            errors.append(f"Inspection PDF not found: {inspection_pdf}")
        else:
            size = Path(inspection_pdf).stat().st_size
            if size < 1024:
                errors.append(f"Inspection PDF too small ({size} bytes), may be corrupted")
        
        if not Path(thermal_pdf).exists():
            errors.append(f"Thermal PDF not found: {thermal_pdf}")
        else:
            size = Path(thermal_pdf).stat().st_size
            if size < 1024:
                errors.append(f"Thermal PDF too small ({size} bytes), may be corrupted")
        
        if errors:
            for error in errors:
                print(f"[ERROR] {error}")
            raise FileNotFoundError("Missing or invalid required input files")
        
        print("[OK] Input files validated")
    
    def handle_extraction_failure(self, doc_type: str, error: Exception, retry_count: int = 0):
        """
        Handle LLM extraction failures gracefully
        
        Args:
            doc_type: Type of document being extracted
            error: The exception that occurred
            retry_count: Number of retries attempted
            
        Raises:
            Exception: Re-raises the error after logging
        """
        print(f"\n[WARNING] {doc_type} extraction failed: {error}")
        
        if retry_count < 2:
            print(f"Attempting retry {retry_count + 1}/2...")
            return True
        else:
            print("[ERROR] Maximum retries exceeded")
            print("Suggestion: Check your API key and rate limits")
            raise  # Re-raise after max retries
    
    def handle_merge_failure(self, error: Exception):
        """
        Handle data merging failures
        
        Args:
            error: The exception that occurred
        """
        print(f"\n[WARNING] Data merging failed: {error}")
        print("This may indicate:")
        print("  • Incompatible data formats")
        print("  • Missing required fields")
        print("  • Embedding model issues")
        raise
    
    def handle_generation_failure(self, error: Exception):
        """
        Handle report generation failures
        
        Args:
            error: The exception that occurred
        """
        print(f"\n[WARNING] Report generation failed: {error}")
        print("This may indicate:")
        print("  • Invalid merged data")
        print("  • DOCX library issues")
        print("  • Insufficient permissions")
        raise
    
    def run(
        self,
        inspection_pdf: str = 'data/input/Sample Report.pdf',
        thermal_pdf: str = 'data/input/Thermal Images.pdf',
        output_path: str = 'data/output/DDR_Report_Final.docx'
    ) -> bool:
        """
        Run the complete DDR generation pipeline
        
        Args:
            inspection_pdf: Path to inspection report PDF
            thermal_pdf: Path to thermal imaging report PDF
            output_path: Path for output DOCX file
            
        Returns:
            bool: True if successful, False otherwise
        """
        
        try:
            # STEP 0: VALIDATE INPUTS
            self.validate_inputs(inspection_pdf, thermal_pdf)
            
            # STEP 1: PARSE DOCUMENTS
            log_step("STEP 1: Document Parsing", "START")
            
            try:
                inspection_doc = self.parser.extract_text_from_pdf(inspection_pdf)
                thermal_doc = self.parser.extract_text_from_pdf(thermal_pdf)
            except Exception as e:
                log_step("STEP 1: Document Parsing", "ERROR", str(e))
                print("\n[ERROR] PDF parsing failed. Possible causes:")
                print("  • Corrupted PDF file")
                print("  • Encrypted/password-protected PDF")
                print("  • Unsupported PDF format")
                raise
            
            log_step(
                "STEP 1: Document Parsing",
                "COMPLETE",
                f"Inspection: {inspection_doc.metadata['num_pages']} pages, "
                f"Thermal: {thermal_doc.metadata['num_pages']} pages"
            )
            
            # Save intermediate results
            self._save_intermediate('parsed_inspection.txt', inspection_doc.text)
            self._save_intermediate('parsed_thermal.txt', thermal_doc.text)
            
            # STEP 2: EXTRACT STRUCTURED DATA
            log_step("STEP 2: LLM Data Extraction", "START")
            
            # Extract with retry logic
            inspection_data = None
            thermal_data = None
            
            for retry in range(3):
                try:
                    if inspection_data is None:
                        inspection_data = self.extractor.extract_inspection_data(inspection_doc.text)
                        import time
                        time.sleep(10)  # Wait between extractions to avoid rate limits
                    if thermal_data is None:
                        thermal_data = self.extractor.extract_thermal_data(thermal_doc.text)
                    break
                except Exception as e:
                    if retry < 2:
                        self.handle_extraction_failure("Data", e, retry)
                        import time
                        time.sleep(2 ** retry)  # Exponential backoff
                    else:
                        raise
            
            log_step(
                "STEP 2: LLM Data Extraction",
                "COMPLETE",
                f"Inspection: {len(inspection_data.observations)} observations, "
                f"Thermal: {len(thermal_data.readings)} readings"
            )
            
            # Save intermediate results
            self._save_intermediate('extracted_inspection.json', inspection_data.model_dump())
            self._save_intermediate('extracted_thermal.json', thermal_data.model_dump())
            
            # STEP 3: CONVERT THERMAL READINGS TO OBSERVATIONS
            log_step("STEP 3: Thermal Data Conversion", "START")
            
            thermal_observations = self._convert_thermal_to_observations(thermal_data)
            
            log_step(
                "STEP 3: Thermal Data Conversion",
                "COMPLETE",
                f"Converted {len(thermal_observations)} thermal readings"
            )
            
            # STEP 4: MERGE AND DEDUPLICATE
            log_step("STEP 4: Data Merging & Deduplication", "START")
            
            try:
                merged_result = self.merger.merge_observations(
                    inspection_data.observations,
                    thermal_observations
                )
            except Exception as e:
                self.handle_merge_failure(e)
            
            log_step(
                "STEP 4: Data Merging & Deduplication",
                "COMPLETE",
                f"Original: {len(inspection_data.observations) + len(thermal_observations)}, "
                f"Merged: {merged_result['total_observations']}, "
                f"Duplicates: {merged_result['duplicate_groups']}"
            )
            
            # Save intermediate results
            self.merger.save_merge_results(merged_result, 'data/intermediate/merged_observations.json')
            
            # STEP 5: GENERATE DDR REPORT
            log_step("STEP 5: DDR Report Generation", "START")
            
            try:
                ddr_document = self.generator.generate_ddr(
                    merged_result,
                    inspection_data.property_details,
                    merged_result['conflicts_detected']
                )
                
                self.generator.save_report(ddr_document, output_path)
            except Exception as e:
                self.handle_generation_failure(e)
            
            log_step(
                "STEP 5: DDR Report Generation",
                "COMPLETE",
                f"Report saved: {output_path}"
            )
            
            # FINAL SUMMARY
            self._print_summary(
                inspection_data,
                thermal_data,
                merged_result,
                output_path
            )
            
            return True
            
        except Exception as e:
            log_step("PIPELINE ERROR", "ERROR", str(e))
            import traceback
            traceback.print_exc()
            return False
    
    def _convert_thermal_to_observations(self, thermal_data) -> list:
        """
        Convert thermal readings to observation format
        """
        observations = []
        
        for reading in thermal_data.readings:
            # Skip if location is not available
            location = reading.location
            if location == "Not Available" or not location:
                location = "Unknown Area"
            
            # Create observation
            obs = Observation(
                location=location,
                issue_type="Thermal Anomaly",
                description=reading.interpretation,
                severity=self._assess_thermal_severity(reading.temperature_delta),
                source_document="thermal",
                confidence=0.8,
                evidence=f"Hotspot: {reading.hotspot_temp}°C, Coldspot: {reading.coldspot_temp}°C, Delta: {reading.temperature_delta}°C"
            )
            observations.append(obs)
        
        return observations
    
    def _assess_thermal_severity(self, temp_delta: float) -> str:
        """
        Assess severity based on temperature differential
        Higher delta = more severe potential issue
        """
        if temp_delta >= 7.0:
            return "High"
        elif temp_delta >= 5.0:
            return "Medium"
        elif temp_delta >= 3.0:
            return "Low"
        else:
            return "Low"
    
    def _save_intermediate(self, filename: str, data):
        """Save intermediate results for debugging"""
        filepath = Path('data/intermediate') / filename
        
        if isinstance(data, str):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(data)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        
        print(f"Saved intermediate: {filepath}")
    
    def _print_summary(
        self,
        inspection_data,
        thermal_data,
        merged_result,
        output_path
    ):
        """Print final execution summary"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*70)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*70)
        print(f"Execution Time: {elapsed:.2f} seconds")
        print()
        print("INPUT:")
        print(f"  • Inspection observations: {len(inspection_data.observations)}")
        print(f"  • Thermal readings: {len(thermal_data.readings)}")
        print(f"  • Total inputs: {len(inspection_data.observations) + len(thermal_data.readings)}")
        print()
        print("PROCESSING:")
        print(f"  • Duplicate groups found: {merged_result['duplicate_groups']}")
        print(f"  • Deduplication rate: {merged_result['metadata']['deduplication_rate']:.1%}")
        print(f"  • Conflicts detected: {len(merged_result['conflicts_detected'])}")
        print()
        print("OUTPUT:")
        print(f"  • Unique observations: {merged_result['total_observations']}")
        print(f"  • Report location: {output_path}")
        
        if Path(output_path).exists():
            print(f"  • File size: {Path(output_path).stat().st_size / 1024:.1f} KB")
        
        print()
        
        # Show breakdown by location
        by_location = {}
        for obs in merged_result['merged_observations']:
            by_location[obs.location] = by_location.get(obs.location, 0) + 1
        
        print("OBSERVATIONS BY AREA:")
        for location, count in sorted(by_location.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  • {location}: {count}")
        
        print()
        print("="*70)
        print("[OK] PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print()
        print(f"Open your report: {output_path}")
        print()


def main():
    """Main entry point"""
    
    # Parse command line arguments (optional)
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate DDR Report from inspection and thermal PDFs'
    )
    parser.add_argument(
        '--inspection',
        default='data/input/Sample Report.pdf',
        help='Path to inspection PDF'
    )
    parser.add_argument(
        '--thermal',
        default='data/input/Thermal Images.pdf',
        help='Path to thermal PDF'
    )
    parser.add_argument(
        '--output',
        default='data/output/DDR_Report_Final.docx',
        help='Output DOCX path'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Configuration file path'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    try:
        pipeline = DDRPipeline(config_path=args.config)
        success = pipeline.run(
            inspection_pdf=args.inspection,
            thermal_pdf=args.thermal,
            output_path=args.output
        )
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
