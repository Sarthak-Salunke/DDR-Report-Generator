"""
Document Parser Module
Handles PDF parsing and text extraction for inspection and thermal reports
"""

import PyPDF2
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class ExtractedDocument:
    """Structured representation of extracted document"""
    text: str
    pages: List[str]
    metadata: Dict
    document_type: str
    extraction_quality: float = field(default=1.0)
    warnings: List[str] = field(default_factory=list)


class DocumentParser:
    """
    Handles PDF parsing and text extraction
    Supports inspection reports and thermal imaging reports
    """
    
    def __init__(self, verbose: bool = True):
        self.supported_formats = ['.pdf']
        self.verbose = verbose
        
    def extract_text_from_pdf(self, pdf_path: str) -> ExtractedDocument:
        """
        Extract text content from PDF file with quality checks
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ExtractedDocument with text, pages, and metadata
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file format not supported
        """
        # Validate input
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {pdf_path.suffix}")
        
        if self.verbose:
            print(f"📄 Parsing: {pdf_path.name}")
        
        # Extract text
        pages = []
        full_text = []
        warnings = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    try:
                        page_text = page.extract_text()
                        
                        # Quality check
                        if not page_text or len(page_text.strip()) < 10:
                            warnings.append(
                                f"Page {page_num}: Very little text extracted (possible image-only page)"
                            )
                        
                        pages.append(page_text)
                        full_text.append(f"--- Page {page_num} ---\n{page_text}")
                        
                    except Exception as e:
                        warnings.append(f"Page {page_num}: Extraction error - {str(e)}")
                        pages.append("")
                
                # Calculate extraction quality
                extraction_quality = self._calculate_quality(pages, warnings)
                
                # Prepare metadata
                metadata = {
                    'num_pages': len(pages),
                    'file_path': str(pdf_path),
                    'file_name': pdf_path.name,
                    'file_size_kb': pdf_path.stat().st_size / 1024,
                    'total_chars': sum(len(p) for p in pages) if pages else 0
                }
                
                # Infer document type
                # We need to construct full text string for type inference
                joined_text = "\n\n".join(full_text)
                doc_type = self._infer_document_type(pdf_path.name, full_text)
                
                if self.verbose:
                    print(f"  ✓ Extracted {len(pages)} pages")
                    print(f"  ✓ Type: {doc_type}")
                    print(f"  ✓ Quality: {extraction_quality:.1%}")
                    if warnings:
                        print(f"  ⚠ {len(warnings)} warnings")
                
                return ExtractedDocument(
                    text=joined_text,
                    pages=pages,
                    metadata=metadata,
                    document_type=doc_type,
                    extraction_quality=extraction_quality,
                    warnings=warnings
                )
                
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF: {str(e)}")
    
    def _calculate_quality(self, pages: List[str], warnings: List[str]) -> float:
        """Calculate extraction quality score (0-1)"""
        if not pages:
            return 0.0
        
        # Factors affecting quality
        total_len = sum(len(p) for p in pages if p)
        avg_page_length = total_len / len(pages) if len(pages) > 0 else 0
        empty_pages = sum(1 for p in pages if not p or len(p.strip()) < 10)
        
        quality = 1.0
        quality -= (empty_pages / len(pages)) * 0.5  # Penalize empty pages
        quality -= (len(warnings) / (len(pages) * 2)) * 0.3  # Penalize warnings
        
        if avg_page_length < 100:
            quality *= 0.7  # Low text content
        
        return max(0.0, min(1.0, quality))
    
    def _infer_document_type(self, filename: str, content: List[str]) -> str:
        """
        Infer document type from filename and content
        
        Returns: 'inspection', 'thermal', or 'unknown'
        """
        filename_lower = filename.lower()
        content_str = " ".join(content).lower()
        
        # Filename-based detection
        if any(keyword in filename_lower for keyword in ['thermal', 'thermograph', 'infrared']):
            return 'thermal'
        elif any(keyword in filename_lower for keyword in ['inspection', 'sample', 'report']):
            return 'inspection'
        
        # Content-based detection
        thermal_keywords = ['hotspot', 'coldspot', 'temperature', 'thermal', '°c']
        inspection_keywords = ['inspection', 'dampness', 'observation', 'defect', 'impacted area']
        
        thermal_score = sum(1 for kw in thermal_keywords if kw in content_str)
        inspection_score = sum(1 for kw in inspection_keywords if kw in content_str)
        
        if thermal_score > inspection_score and thermal_score >= 3:
            return 'thermal'
        elif inspection_score > thermal_score and inspection_score >= 3:
            return 'inspection'
        
        return 'unknown'
    
    def extract_both_documents(
        self,
        inspection_path: str,
        thermal_path: str
    ) -> Tuple[ExtractedDocument, ExtractedDocument]:
        """
        Convenience method to extract both documents
        
        Returns:
            (inspection_doc, thermal_doc)
        """
        inspection_doc = self.extract_text_from_pdf(inspection_path)
        thermal_doc = self.extract_text_from_pdf(thermal_path)
        
        # Verify types
        if inspection_doc.document_type not in ['inspection', 'unknown']:
            print(f"⚠ Warning: {inspection_path} may not be an inspection report")
        
        if thermal_doc.document_type not in ['thermal', 'unknown']:
            print(f"⚠ Warning: {thermal_path} may not be a thermal report")
        
        return inspection_doc, thermal_doc


# Utility Functions

def clean_extracted_text(text: str) -> str:
    """Clean extracted text by removing excessive whitespace and artifacts"""
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove page number artifacts
    text = re.sub(r'Page \d+ of \d+', '', text)
    
    return text.strip()


def extract_tables_from_text(text: str) -> List[str]:
    """
    Attempt to extract table-like structures from text
    (Simple heuristic-based approach)
    """
    tables = []
    lines = text.split('\n')
    
    current_table = []
    in_table = False
    
    for line in lines:
        # Detect table by presence of multiple tab/space-separated values
        if '\t' in line or re.search(r'\s{2,}', line):
            current_table.append(line)
            in_table = True
        else:
            if in_table and current_table:
                tables.append('\n'.join(current_table))
                current_table = []
            in_table = False
    
    return tables



def test_parser():
    """Test the document parser with sample files"""
    parser = DocumentParser(verbose=True)
    
    # Use absolute paths or check relative to execution
    base_dir = Path("e:/Project/ddr_report_generator")
    sample_report = base_dir / 'data/input/Sample Report.pdf'
    thermal_images = base_dir / 'data/input/Thermal Images.pdf'
    
    try:
        # Test inspection report
        print("\n" + "="*60)
        print(f"Testing Inspection Report Parsing: {sample_report}")
        print("="*60)
        
        if not sample_report.exists():
            print(f"❌ File not found: {sample_report}")
        else:
            inspection_doc = parser.extract_text_from_pdf(str(sample_report))
            print(f"\nExtracted {len(inspection_doc.pages)} pages")
            print(f"Total characters: {len(inspection_doc.text)}")
            print(f"Quality score: {inspection_doc.extraction_quality:.1%}")
        
        # Test thermal report
        print("\n" + "="*60)
        print(f"Testing Thermal Report Parsing: {thermal_images}")
        print("="*60)
        
        if not thermal_images.exists():
            print(f"❌ File not found: {thermal_images}")
        else:
            thermal_doc = parser.extract_text_from_pdf(str(thermal_images))
            print(f"\nExtracted {len(thermal_doc.pages)} pages")
            print(f"Total characters: {len(thermal_doc.text)}")
            print(f"Quality score: {thermal_doc.extraction_quality:.1%}")
        
    except Exception as e:
        print(f"\n❌ Parser test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_parser()
