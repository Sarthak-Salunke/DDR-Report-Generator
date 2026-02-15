"""
DDR Generator Module
Creates professional DOCX reports from merged observations
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from src.data_models import MergedObservation, PropertyDetails
from prompts.generation_prompts import (
    build_executive_summary_prompt,
    build_recommendations_prompt,
    build_conflict_note_prompt
)


class DDRGenerator:
    """
    Generates professional Diagnostic Defect Report (DDR) in DOCX format
    """
    
    def __init__(self, llm_manager=None, verbose: bool = True):
        """
        Initialize DDR generator
        
        Args:
            llm_manager: LLM manager for content generation
            verbose: Print progress messages
        """
        self.llm_manager = llm_manager
        self.verbose = verbose
    
    def generate_ddr(
        self,
        merged_result: Dict,
        property_details: PropertyDetails,
        conflicts: Optional[List[Dict]] = None
    ) -> Document:
        """
        Generate complete DDR document
        
        Args:
            merged_result: Result from DataMerger
            property_details: Property information
            conflicts: List of detected conflicts
            
        Returns:
            python-docx Document object
        """
        if conflicts is None:
            conflicts = merged_result.get('conflicts_detected', [])
        
        if self.verbose:
            print("\n📝 Generating DDR Report...")
        
        doc = Document()
        
        # Set up document styles
        self._setup_styles(doc)
        
        # 1. Title Page
        self._add_title_page(doc, property_details)
        
        # 2. Executive Summary
        self._add_executive_summary(doc, merged_result, property_details)
        
        # 3. Property Details
        self._add_property_details(doc, property_details)
        
        # 4. Observations by Area
        self._add_observations_by_area(doc, merged_result['merged_observations'])
        
        # 5. Conflicts (if any)
        if conflicts:
            self._add_conflicts_section(doc, conflicts)
        
        # 6. Recommendations
        self._add_recommendations(doc, merged_result)
        
        # 7. Limitations
        self._add_limitations(doc)
        
        if self.verbose:
            print("✅ DDR Report generated successfully!")
        
        return doc
    
    def save_report(self, doc: Document, output_path: str):
        """
        Save report to file
        
        Args:
            doc: Document object
            output_path: Path to save file
        """
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save document
        doc.save(output_path)
        
        if self.verbose:
            print(f"💾 Report saved to: {output_path}")
    
    def _setup_styles(self, doc: Document):
        """Configure document styles"""
        styles = doc.styles
        
        # Heading 1
        heading1 = styles['Heading 1']
        heading1.font.size = Pt(16)
        heading1.font.bold = True
        heading1.font.color.rgb = RGBColor(31, 78, 121)
        
        # Heading 2
        heading2 = styles['Heading 2']
        heading2.font.size = Pt(14)
        heading2.font.bold = True
        heading2.font.color.rgb = RGBColor(68, 114, 196)
        
        # Normal
        normal = styles['Normal']
        normal.font.name = 'Calibri'
        normal.font.size = Pt(11)
    
    def _add_title_page(self, doc: Document, property_details: PropertyDetails):
        """Add title page"""
        # Title
        title = doc.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = title.add_run("DIAGNOSTIC DEFECT REPORT")
        run.font.size = Pt(24)
        run.font.bold = True
        run.font.color.rgb = RGBColor(31, 78, 121)
        
        doc.add_paragraph()  # Space
        
        # Property type
        prop_type = doc.add_paragraph()
        prop_type.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = prop_type.add_run(f"Property Type: {property_details.property_type}")
        run.font.size = Pt(14)
        
        # Date
        date = doc.add_paragraph()
        date.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = date.add_run(f"Report Generated: {datetime.now().strftime('%B %d, %Y')}")
        run.font.size = Pt(12)
        
        doc.add_page_break()
    
    def _add_executive_summary(
        self,
        doc: Document,
        merged_result: Dict,
        property_details: PropertyDetails
    ):
        """Add executive summary section"""
        doc.add_heading('Executive Summary', level=1)
        
        # Generate summary using LLM if available
        if self.llm_manager:
            try:
                summary_text = self._generate_executive_summary(
                    merged_result,
                    property_details
                )
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  LLM summary generation failed: {e}")
                summary_text = self._create_default_summary(merged_result, property_details)
        else:
            summary_text = self._create_default_summary(merged_result, property_details)
        
        doc.add_paragraph(summary_text)
        doc.add_paragraph()  # Space
    
    def _generate_executive_summary(self, merged_result: Dict, property_details: PropertyDetails) -> str:
        """Generate summary using LLM"""
        # Count observations by severity
        observations = merged_result['merged_observations']
        severity_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
        
        for obs in observations:
            if obs.severity:
                severity_counts[obs.severity] = severity_counts.get(obs.severity, 0) + 1
        
        # Get major findings (top 3-5 issues)
        major_findings = [
            f"{obs.issue_type} in {obs.location}"
            for obs in sorted(observations, key=lambda x: x.confidence, reverse=True)[:5]
        ]
        
        prompt = build_executive_summary_prompt(
            property_details=property_details.model_dump(),
            observations_summary={
                'total': len(observations),
                'critical': severity_counts.get('Critical', 0),
                'high': severity_counts.get('High', 0),
                'medium': severity_counts.get('Medium', 0),
                'low': severity_counts.get('Low', 0)
            },
            major_findings=major_findings
        )
        
        # Generate with LLM
        summary = self.llm_manager.generate(prompt, task="executive_summary")
        return summary.strip()
    
    def _create_default_summary(self, merged_result: Dict, property_details: PropertyDetails) -> str:
        """Create default summary without LLM"""
        total = merged_result['total_observations']
        dedup_rate = merged_result['metadata']['deduplication_rate']
        
        return f"""This Diagnostic Defect Report presents findings from a comprehensive inspection of the {property_details.property_type} conducted on {property_details.inspection_date}. 

A total of {total} distinct defects were identified through visual inspection and thermal imaging analysis. The inspection covered multiple areas including interior rooms, bathrooms, kitchen, and external elements.

The most commonly observed issues include dampness at skirting levels, bathroom tile defects, and external wall conditions. All findings have been documented with photographic evidence and thermal readings where applicable.

This report provides detailed observations, root cause analysis, and recommendations for remedial action."""
    
    def _add_property_details(self, doc: Document, property_details: PropertyDetails):
        """Add property details section"""
        doc.add_heading('Property Information', level=1)
        
        # Create table
        table = doc.add_table(rows=5, cols=2)
        table.style = 'Light Grid Accent 1'
        
        # Add data
        rows_data = [
            ('Property Type', property_details.property_type),
            ('Inspection Date', property_details.inspection_date),
            ('Inspected By', ', '.join(property_details.inspected_by) if property_details.inspected_by else 'Not Available'),
            ('Number of Floors', str(property_details.floors) if property_details.floors else 'Not Available'),
            ('Overall Score', f"{property_details.score}%" if property_details.score else 'Not Available')
        ]
        
        for i, (label, value) in enumerate(rows_data):
            row = table.rows[i]
            row.cells[0].text = label
            row.cells[0].paragraphs[0].runs[0].font.bold = True
            row.cells[1].text = value
        
        doc.add_paragraph()  # Space
    
    def _add_observations_by_area(self, doc: Document, observations: List[MergedObservation]):
        """Add observations organized by area"""
        doc.add_heading('Detailed Observations', level=1)
        
        # Group by location
        by_area = {}
        for obs in observations:
            area = obs.location
            if area not in by_area:
                by_area[area] = []
            by_area[area].append(obs)
        
        # Add each area
        for area, area_obs in sorted(by_area.items()):
            doc.add_heading(area, level=2)
            
            for i, obs in enumerate(area_obs, 1):
                # Observation header
                p = doc.add_paragraph()
                p.add_run(f"{i}. {obs.issue_type}").bold = True
                
                # Description
                doc.add_paragraph(obs.description, style='List Bullet')
                
                # Severity
                if obs.severity:
                    severity_p = doc.add_paragraph()
                    severity_p.add_run("Severity: ").bold = True
                    run = severity_p.add_run(obs.severity)
                    
                    # Color code severity
                    if obs.severity == "Critical":
                        run.font.color.rgb = RGBColor(192, 0, 0)
                    elif obs.severity == "High":
                        run.font.color.rgb = RGBColor(255, 102, 0)
                    elif obs.severity == "Medium":
                        run.font.color.rgb = RGBColor(255, 192, 0)
                
                # Sources
                sources_p = doc.add_paragraph()
                sources_p.add_run("Sources: ").bold = True
                sources_p.add_run(', '.join(obs.sources))
                
                # Evidence
                if obs.evidence:
                    evidence_p = doc.add_paragraph()
                    evidence_p.add_run("Evidence: ").bold = True
                    evidence_p.add_run(', '.join(obs.evidence))
                
                # Conflict note if any
                if obs.conflict_detected:
                    conflict_p = doc.add_paragraph()
                    run = conflict_p.add_run("⚠️ Note: ")
                    run.bold = True
                    run.font.color.rgb = RGBColor(192, 0, 0)
                    conflict_p.add_run(obs.conflict_details or "Conflicting information detected from multiple sources")
                
                doc.add_paragraph()  # Space
    
    def _add_conflicts_section(self, doc: Document, conflicts: List[Dict]):
        """Add conflicts section"""
        doc.add_page_break()
        doc.add_heading('Data Conflicts Requiring Review', level=1)
        
        doc.add_paragraph(
            "The following conflicts were detected during data merging. "
            "These require additional verification or professional judgment."
        )
        
        for i, conflict in enumerate(conflicts, 1):
            doc.add_heading(f"Conflict {i}: {conflict['location']}", level=2)
            
            conflict_p = doc.add_paragraph()
            conflict_p.add_run("Issue Type: ").bold = True
            conflict_p.add_run(conflict['issue_type'])
            
            details_p = doc.add_paragraph()
            details_p.add_run("Details: ").bold = True
            details_p.add_run(conflict['details'])
            
            sources_p = doc.add_paragraph()
            sources_p.add_run("Sources: ").bold = True
            sources_p.add_run(', '.join(conflict['sources']))
            
            action_p = doc.add_paragraph()
            action_p.add_run("Recommended Action: ").bold = True
            action_p.add_run(conflict.get('action_required', 'Further investigation recommended'))
            
            doc.add_paragraph()  # Space
    
    def _add_recommendations(self, doc: Document, merged_result: Dict):
        """Add recommendations section"""
        doc.add_page_break()
        doc.add_heading('Recommendations', level=1)
        
        # Generate recommendations using LLM if available
        if self.llm_manager:
            try:
                recommendations = self._generate_recommendations(merged_result)
                doc.add_paragraph(recommendations)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  LLM recommendations generation failed: {e}")
                self._add_default_recommendations(doc, merged_result)
        else:
            self._add_default_recommendations(doc, merged_result)
    
    def _generate_recommendations(self, merged_result: Dict) -> str:
        """Generate recommendations using LLM"""
        observations = merged_result['merged_observations']
        
        # Create summary
        summary = f"Total issues: {len(observations)}\n"
        summary += "Key areas affected: " + ', '.join(set(obs.location for obs in observations[:10]))
        
        prompt = build_recommendations_prompt(
            observations_summary=summary,
            root_causes=[]  # Add if available
        )
        
        # Generate with LLM
        recommendations = self.llm_manager.generate(prompt, task="recommendations")
        return recommendations.strip()
    
    def _add_default_recommendations(self, doc: Document, merged_result: Dict):
        """Add default recommendations without LLM"""
        doc.add_heading('Immediate Actions', level=2)
        doc.add_paragraph("Address all plumbing leaks and damaged pipe joints", style='List Bullet')
        doc.add_paragraph("Repair or replace hollow/damaged bathroom tiles", style='List Bullet')
        doc.add_paragraph("Fix external wall cracks and ensure proper waterproofing", style='List Bullet')
        
        doc.add_heading('Short-term Actions (1-3 months)', level=2)
        doc.add_paragraph("Conduct comprehensive waterproofing of affected areas", style='List Bullet')
        doc.add_paragraph("Address dampness issues with proper ventilation solutions", style='List Bullet')
        doc.add_paragraph("Monitor resolved areas for recurrence", style='List Bullet')
        
        doc.add_heading('Long-term Considerations', level=2)
        doc.add_paragraph("Implement regular maintenance schedule", style='List Bullet')
        doc.add_paragraph("Consider periodic thermal imaging inspections", style='List Bullet')
        doc.add_paragraph("Maintain records of all repairs and interventions", style='List Bullet')
    
    def _add_limitations(self, doc: Document):
        """Add limitations section"""
        doc.add_page_break()
        doc.add_heading('Limitations & Disclaimers', level=1)
        
        limitations = [
            "This report is based on visual inspection and thermal imaging data available at the time of inspection.",
            "No destructive or invasive testing was performed.",
            "Hidden defects not visible during inspection may exist.",
            "This report does not constitute a structural engineering assessment.",
            "Further investigation by specialized professionals may be required for specific issues.",
            "Recommendations are based on observations made and may require adjustment based on additional findings.",
            "This report is valid as of the inspection date and conditions may change over time."
        ]
        
        for limitation in limitations:
            doc.add_paragraph(limitation, style='List Bullet')
        
        doc.add_paragraph()
        
        # Footer
        footer_p = doc.add_paragraph()
        footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer_p.add_run("--- End of Report ---")
        run.italic = True
        run.font.size = Pt(10)



def test_generator():
    """Test DDR generator with sample data"""
    from src.data_models import MergedObservation, PropertyDetails
    
    print("="*60)
    print("Testing DDR Generator")
    print("="*60)
    
    # Sample property details
    property_details = PropertyDetails(
        property_type="Flat",
        floors=11,
        inspection_date="27.09.2022",
        inspected_by=["Krushna", "Mahesh"],
        score=85.71,
        flagged_items=1
    )
    
    # Sample observations
    observations = [
        MergedObservation(
            location="Hall",
            issue_type="Dampness",
            description="Dampness observed at skirting level",
            severity="Medium",
            sources=["inspection", "thermal"],
            evidence=["Photo 1-7", "Temp delta: 5.4°C"],
            confidence=0.88,
            is_duplicate=True
        ),
        MergedObservation(
            location="Kitchen",
            issue_type="Crack",
            description="Small crack on wall surface",
            severity="Low",
            sources=["inspection"],
            evidence=["Photo 31"],
            confidence=0.80
        )
    ]
    
    # Sample merged result
    merged_result = {
        'merged_observations': observations,
        'total_observations': len(observations),
        'duplicate_groups': 1,
        'conflicts_detected': [],
        'metadata': {
            'deduplication_rate': 0.33
        }
    }
    
    # Generate DDR
    generator = DDRGenerator(llm_manager=None, verbose=True)
    doc = generator.generate_ddr(merged_result, property_details, [])
    
    # Save
    output_path = 'data/output/DDR_Test_Report.docx'
    generator.save_report(doc, output_path)
    
    print(f"\n✅ Test report generated: {output_path}")
    print("="*60)


if __name__ == "__main__":
    test_generator()
