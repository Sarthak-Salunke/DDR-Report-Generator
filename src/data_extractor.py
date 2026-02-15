"""
Data Extractor Module
Uses multi-provider LLM system to extract structured data from documents
"""

import json
import os
import yaml
from typing import Dict, List, Optional
from datetime import datetime

from src.data_models import (
    InspectionData, ThermalData, PropertyDetails,
    Observation, ThermalReading
)
from prompts.extraction_prompts import (
    build_inspection_prompt,
    build_thermal_prompt,
    build_validation_prompt
)
from src.llm_manager import LLMManager


class DataExtractor:
    """
    Handles LLM-based extraction of structured data from documents
    Uses multi-provider system (Gemini primary, Groq backup)
    """
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        verbose: bool = True
    ):
        """
        Initialize DataExtractor with multi-provider LLM system
        
        Args:
            config_path: Path to configuration file
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize LLM Manager
        self.llm_manager = LLMManager(
            config=self.config['llm'],
            verbose=verbose
        )
        
        if self.verbose:
            print(f"✓ DataExtractor initialized with multi-provider LLM system")
    
    def extract_inspection_data(self, document_text: str) -> InspectionData:
        """
        Extract structured inspection data from inspection report
        
        Args:
            document_text: Full text of inspection report
            
        Returns:
            InspectionData object with validated structured data
            
        Raises:
            ValueError: If extraction fails or data invalid
        """
        if self.verbose:
            print("\n🧠 Extracting inspection data with LLM...")
        
        prompt = build_inspection_prompt(document_text)
        
        # Call LLM via manager (automatic failover)
        try:
            data_dict = self.llm_manager.generate_json(
                prompt=prompt,
                task="inspection_extraction"
            )
            
            if self.verbose:
                print(f"  ✓ LLM response received")
            
            # Convert to Pydantic models
            property_details = PropertyDetails(**data_dict['property_details'])
            
            observations = []
            for obs_dict in data_dict['observations']:
                obs = Observation(**obs_dict)
                observations.append(obs)
            
            root_causes = data_dict.get('root_causes', [])
            metadata = data_dict.get('metadata', {})
            
            inspection_data = InspectionData(
                property_details=property_details,
                observations=observations,
                root_causes=root_causes,
                metadata=metadata
            )
            
            if self.verbose:
                print(f"  ✓ Extracted {len(observations)} observations")
                print(f"  ✓ Identified {len(root_causes)} root causes")
            
            return inspection_data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {str(e)}")
        except Exception as e:
            raise ValueError(f"Extraction failed: {str(e)}")
    
    def extract_thermal_data(self, document_text: str) -> ThermalData:
        """
        Extract structured thermal imaging data from thermal report
        
        Args:
            document_text: Full text of thermal imaging report
            
        Returns:
            ThermalData object with validated readings
            
        Raises:
            ValueError: If extraction fails or data invalid
        """
        if self.verbose:
            print("\n🌡️  Extracting thermal data with LLM...")
        
        prompt = build_thermal_prompt(document_text)
        
        # Call LLM via manager (automatic failover)
        try:
            data_dict = self.llm_manager.generate_json(
                prompt=prompt,
                task="thermal_extraction"
            )
            
            if self.verbose:
                print(f"  ✓ LLM response received")
            
            # Convert to Pydantic models
            readings = []
            for reading_dict in data_dict['readings']:
                reading = ThermalReading(**reading_dict)
                readings.append(reading)
            
            thermal_data = ThermalData(
                readings=readings,
                date=data_dict.get('date', 'Not Available'),
                device=data_dict.get('device'),
                metadata=data_dict.get('metadata', {})
            )
            
            if self.verbose:
                print(f"  ✓ Extracted {len(readings)} thermal readings")
            
            return thermal_data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {str(e)}")
        except Exception as e:
            raise ValueError(f"Extraction failed: {str(e)}")
    
    def validate_extraction(
        self,
        source_document: str,
        extracted_data: Dict
    ) -> Dict:
        """
        Validate extracted data against source document
        
        Args:
            source_document: Original document text
            extracted_data: Extracted data as dictionary
            
        Returns:
            Validation report dictionary
        """
        if self.verbose:
            print("\n✓ Validating extraction accuracy...")
        
        # Build validation prompt
        prompt = build_validation_prompt(
            source_document,
            json.dumps(extracted_data, indent=2)
        )
        
        try:
            validation_report = self.llm_manager.generate_json(
                prompt=prompt,
                task="validation"
            )
            
            if self.verbose:
                if validation_report['is_valid']:
                    print(f"  ✅ Validation passed (score: {validation_report['accuracy_score']:.2f})")
                else:
                    print(f"  ⚠️  Validation issues found:")
                    for error in validation_report['errors_found']:
                        print(f"    - {error['type']}: {error['description']}")
            
            return validation_report
            
        except Exception as e:
            return {
                'is_valid': False,
                'accuracy_score': 0.0,
                'errors_found': [{'type': 'system_error', 'description': str(e)}],
                'warnings': [],
                'summary': f'Validation failed: {str(e)}'
            }
    

    
    def batch_extract(
        self,
        documents: List[tuple[str, str]]
    ) -> List[tuple[InspectionData, ThermalData]]:
        """
        Extract data from multiple document pairs
        
        Args:
            documents: List of (inspection_text, thermal_text) tuples
            
        Returns:
            List of (InspectionData, ThermalData) tuples
        """
        results = []
        
        for i, (inspection_text, thermal_text) in enumerate(documents, 1):
            if self.verbose:
                print(f"\n📄 Processing document pair {i}/{len(documents)}")
            
            inspection_data = self.extract_inspection_data(inspection_text)
            thermal_data = self.extract_thermal_data(thermal_text)
            
            results.append((inspection_data, thermal_data))
        
        return results



def test_extractor():
    """Test the data extractor with sample documents"""
    from dotenv import load_dotenv
    from src.document_parser import DocumentParser
    
    load_dotenv()
    
    print("="*60)
    print("Testing Data Extractor with Multi-Provider LLM")
    print("="*60)
    
    # Parse documents
    parser = DocumentParser(verbose=True)
    inspection_doc = parser.extract_text_from_pdf('data/input/Sample_Report.pdf')
    thermal_doc = parser.extract_text_from_pdf('data/input/Thermal_Images.pdf')
    
    # Extract data (uses config.yaml for provider settings)
    extractor = DataExtractor(config_path="config.yaml", verbose=True)
    
    try:
        # Extract inspection data
        inspection_data = extractor.extract_inspection_data(inspection_doc.text)
        print(f"\n✅ Inspection extraction successful:")
        print(f"   - Property: {inspection_data.property_details.property_type}")
        print(f"   - Observations: {len(inspection_data.observations)}")
        print(f"   - Root causes: {len(inspection_data.root_causes)}")
        
        # Extract thermal data
        thermal_data = extractor.extract_thermal_data(thermal_doc.text)
        print(f"\n✅ Thermal extraction successful:")
        print(f"   - Readings: {len(thermal_data.readings)}")
        print(f"   - Date: {thermal_data.date}")
        
        # Validate
        validation = extractor.validate_extraction(
            inspection_doc.text,
            inspection_data.model_dump()
        )
        print(f"\n✅ Validation completed:")
        print(f"   - Valid: {validation['is_valid']}")
        print(f"   - Score: {validation['accuracy_score']:.2f}")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_extractor()
