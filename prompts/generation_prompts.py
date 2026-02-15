"""
Generation prompts for creating DDR report content
Uses LLM to generate professional, client-friendly text
"""

# Executive Summary Generation

EXECUTIVE_SUMMARY_PROMPT = """You are a professional building inspection report writer. Create a concise executive summary for a Diagnostic Defect Report (DDR).

## PROPERTY INFORMATION:
{property_details}

## OBSERVATIONS SUMMARY:
Total issues identified: {total_observations}
Critical issues: {critical_count}
High priority issues: {high_count}
Medium priority issues: {medium_count}
Low priority issues: {low_count}

## MAJOR FINDINGS:
{major_findings}

## YOUR TASK:
Write a professional 3-4 paragraph executive summary that:
1. Opens with property overview
2. Summarizes key findings
3. Highlights critical issues requiring immediate attention
4. Provides overall assessment

**CRITICAL RULES:**
- Use professional, client-friendly language
- Avoid technical jargon where possible
- Be factual and objective
- Do NOT exaggerate or minimize issues
- Keep tone neutral and informative
- Length: 200-300 words

Return ONLY the executive summary text, no additional formatting.
"""

# Recommendations Generation

RECOMMENDATIONS_PROMPT = """You are a professional building inspector providing recommendations based on identified defects.

## OBSERVATIONS:
{observations_summary}

## ROOT CAUSES:
{root_causes}

## YOUR TASK:
Generate 5-8 professional recommendations organized by priority:

**IMMEDIATE ACTIONS (within 1 week):**
- [Actions for critical/high priority issues]

**SHORT-TERM ACTIONS (within 1 month):**
- [Actions for medium priority issues]

**LONG-TERM CONSIDERATIONS:**
- [Preventive measures and monitoring]

**CRITICAL RULES:**
- Be specific and actionable
- Prioritize by severity and urgency
- Include cost-consciousness where appropriate
- Recommend professional involvement where needed
- Use clear, client-friendly language

Return as a bulleted list with clear priority sections.
"""

# Conflict Resolution Note

CONFLICT_NOTE_PROMPT = """You are a technical writer explaining data discrepancies in a professional report.

## CONFLICT DETAILS:
Location: {location}
Issue Type: {issue_type}
Conflict: {conflict_details}
Sources: {sources}

## YOUR TASK:
Write a brief, professional note (2-3 sentences) explaining this conflict to the client.

**GUIDELINES:**
- Be transparent about the discrepancy
- Suggest possible reasons (different observation times, methods, etc.)
- Recommend further investigation if warranted
- Maintain professional, reassuring tone

Return ONLY the note text.
"""


AREA_DESCRIPTION_PROMPT = """Generate a brief description of issues in a specific area.

## AREA: {area_name}
## ISSUES FOUND:
{issues_list}

Write 2-3 sentences summarizing the condition of this area.
Be factual, professional, and client-friendly.
"""

# Observation Enhancement

OBSERVATION_ENHANCEMENT_PROMPT = """You are a professional building inspector. Enhance this observation for a client report.

## ORIGINAL OBSERVATION:
Location: {location}
Issue: {issue_type}
Description: {description}
Severity: {severity}

## YOUR TASK:
Rewrite the description to be:
- Clear and professional
- Client-friendly (avoid excessive jargon)
- Factual and objective
- Appropriately detailed (2-3 sentences)

**CRITICAL RULES:**
- Maintain the original meaning and severity
- Do NOT add information not in the original
- Do NOT change the severity assessment
- Keep it concise but informative

Return ONLY the enhanced description text.
"""

# Root Cause Analysis

ROOT_CAUSE_ANALYSIS_PROMPT = """You are a building diagnostics expert. Analyze the root causes of observed defects.

## OBSERVATIONS:
{observations}

## YOUR TASK:
Identify 3-5 likely root causes for these issues. For each cause:
1. State the root cause clearly
2. Explain which observations it relates to
3. Suggest investigation or remediation approach

**GUIDELINES:**
- Focus on underlying causes, not symptoms
- Be specific but avoid over-speculation
- Consider common building defect patterns
- Use professional but accessible language

Return as a numbered list.
"""

# Priority Assessment

PRIORITY_ASSESSMENT_PROMPT = """You are a building inspector assessing repair priorities.

## OBSERVATIONS:
{observations}

## YOUR TASK:
Categorize these observations into priority levels and explain the reasoning:

**CRITICAL (Immediate attention required):**
- [Issues posing safety risks or rapid deterioration]

**HIGH (Address within 1 month):**
- [Significant defects requiring prompt attention]

**MEDIUM (Address within 3-6 months):**
- [Moderate issues that should be monitored]

**LOW (Monitor and plan):**
- [Minor issues for future consideration]

For each item, briefly explain WHY it's at that priority level.
"""


TECHNICAL_SUMMARY_PROMPT = """You are a building surveyor creating a technical summary.

## INSPECTION DATA:
Property Type: {property_type}
Inspection Date: {inspection_date}
Total Observations: {total_observations}
Methods Used: {methods}

## OBSERVATIONS BY CATEGORY:
{observations_by_category}

## YOUR TASK:
Write a 2-3 paragraph technical summary covering:
1. Inspection methodology
2. Overall findings and patterns
3. Technical assessment of property condition

Use professional terminology appropriate for technical stakeholders.
"""

# Client-Friendly Explanation

CLIENT_EXPLANATION_PROMPT = """You are explaining a technical building issue to a non-technical client.

## TECHNICAL ISSUE:
{technical_description}

## YOUR TASK:
Rewrite this in simple, client-friendly language that:
- Explains WHAT the issue is
- Explains WHY it matters
- Suggests WHAT should be done

Use analogies or simple comparisons if helpful.
Avoid jargon. Keep it reassuring but honest.
Length: 2-3 sentences.
"""


def build_executive_summary_prompt(
    property_details: dict,
    observations_summary: dict,
    major_findings: list
) -> str:
    """
    Build prompt for executive summary generation
    
    Args:
        property_details: Dictionary with property information
        observations_summary: Dictionary with observation counts by severity
        major_findings: List of major finding descriptions
        
    Returns:
        Formatted prompt string
    """
    # Format property details
    property_text = "\n".join(
        f"{key.replace('_', ' ').title()}: {value}"
        for key, value in property_details.items()
    )
    
    return EXECUTIVE_SUMMARY_PROMPT.format(
        property_details=property_text,
        total_observations=observations_summary.get('total', 0),
        critical_count=observations_summary.get('critical', 0),
        high_count=observations_summary.get('high', 0),
        medium_count=observations_summary.get('medium', 0),
        low_count=observations_summary.get('low', 0),
        major_findings='\n'.join(f"- {finding}" for finding in major_findings)
    )


def build_recommendations_prompt(
    observations_summary: str,
    root_causes: list
) -> str:
    """
    Build prompt for recommendations generation
    
    Args:
        observations_summary: Text summary of observations
        root_causes: List of identified root causes
        
    Returns:
        Formatted prompt string
    """
    return RECOMMENDATIONS_PROMPT.format(
        observations_summary=observations_summary,
        root_causes='\n'.join(f"- {cause}" for cause in root_causes)
    )


def build_conflict_note_prompt(conflict: dict) -> str:
    """
    Build prompt for conflict note generation
    
    Args:
        conflict: Dictionary with conflict details
        
    Returns:
        Formatted prompt string
    """
    return CONFLICT_NOTE_PROMPT.format(
        location=conflict.get('location', 'Unknown'),
        issue_type=conflict.get('issue_type', 'Unknown'),
        conflict_details=conflict.get('details', 'Discrepancy detected'),
        sources=', '.join(conflict.get('sources', []))
    )


def build_area_description_prompt(area_name: str, issues: list) -> str:
    """
    Build prompt for area description
    
    Args:
        area_name: Name of the area/location
        issues: List of issue descriptions
        
    Returns:
        Formatted prompt string
    """
    issues_text = '\n'.join(f"- {issue}" for issue in issues)
    return AREA_DESCRIPTION_PROMPT.format(
        area_name=area_name,
        issues_list=issues_text
    )


def build_observation_enhancement_prompt(observation: dict) -> str:
    """
    Build prompt for enhancing observation description
    
    Args:
        observation: Dictionary with observation details
        
    Returns:
        Formatted prompt string
    """
    return OBSERVATION_ENHANCEMENT_PROMPT.format(
        location=observation.get('location', 'Unknown'),
        issue_type=observation.get('issue_type', 'Unknown'),
        description=observation.get('description', ''),
        severity=observation.get('severity', 'Unknown')
    )


def build_root_cause_analysis_prompt(observations: list) -> str:
    """
    Build prompt for root cause analysis
    
    Args:
        observations: List of observation dictionaries
        
    Returns:
        Formatted prompt string
    """
    obs_text = '\n'.join(
        f"- {obs.get('location', 'Unknown')}: {obs.get('description', '')}"
        for obs in observations
    )
    return ROOT_CAUSE_ANALYSIS_PROMPT.format(observations=obs_text)


def build_priority_assessment_prompt(observations: list) -> str:
    """
    Build prompt for priority assessment
    
    Args:
        observations: List of observation dictionaries
        
    Returns:
        Formatted prompt string
    """
    obs_text = '\n'.join(
        f"- {obs.get('location', 'Unknown')}: {obs.get('description', '')} (Current: {obs.get('severity', 'Unknown')})"
        for obs in observations
    )
    return PRIORITY_ASSESSMENT_PROMPT.format(observations=obs_text)


def build_technical_summary_prompt(
    property_type: str,
    inspection_date: str,
    total_observations: int,
    methods: list,
    observations_by_category: dict
) -> str:
    """
    Build prompt for technical summary
    
    Args:
        property_type: Type of property inspected
        inspection_date: Date of inspection
        total_observations: Total number of observations
        methods: List of inspection methods used
        observations_by_category: Dictionary of observations grouped by category
        
    Returns:
        Formatted prompt string
    """
    methods_text = ', '.join(methods)
    category_text = '\n'.join(
        f"{category}: {count} observations"
        for category, count in observations_by_category.items()
    )
    
    return TECHNICAL_SUMMARY_PROMPT.format(
        property_type=property_type,
        inspection_date=inspection_date,
        total_observations=total_observations,
        methods=methods_text,
        observations_by_category=category_text
    )


def build_client_explanation_prompt(technical_description: str) -> str:
    """
    Build prompt for client-friendly explanation
    
    Args:
        technical_description: Technical description to simplify
        
    Returns:
        Formatted prompt string
    """
    return CLIENT_EXPLANATION_PROMPT.format(
        technical_description=technical_description
    )



def validate_prompt_inputs(**kwargs) -> bool:
    """
    Validate that required inputs for prompts are present
    
    Args:
        **kwargs: Prompt input parameters
        
    Returns:
        True if valid, raises ValueError if not
    """
    for key, value in kwargs.items():
        if value is None:
            raise ValueError(f"Required prompt input '{key}' is None")
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"Required prompt input '{key}' is empty")
        if isinstance(value, (list, dict)) and not value:
            raise ValueError(f"Required prompt input '{key}' is empty")
    
    return True


# Prompt Templates Registry

PROMPT_TEMPLATES = {
    'executive_summary': EXECUTIVE_SUMMARY_PROMPT,
    'recommendations': RECOMMENDATIONS_PROMPT,
    'conflict_note': CONFLICT_NOTE_PROMPT,
    'area_description': AREA_DESCRIPTION_PROMPT,
    'observation_enhancement': OBSERVATION_ENHANCEMENT_PROMPT,
    'root_cause_analysis': ROOT_CAUSE_ANALYSIS_PROMPT,
    'priority_assessment': PRIORITY_ASSESSMENT_PROMPT,
    'technical_summary': TECHNICAL_SUMMARY_PROMPT,
    'client_explanation': CLIENT_EXPLANATION_PROMPT
}


def get_prompt_template(template_name: str) -> str:
    """
    Get a prompt template by name
    
    Args:
        template_name: Name of the template
        
    Returns:
        Prompt template string
        
    Raises:
        KeyError: If template name not found
    """
    if template_name not in PROMPT_TEMPLATES:
        available = ', '.join(PROMPT_TEMPLATES.keys())
        raise KeyError(
            f"Unknown prompt template: '{template_name}'. "
            f"Available templates: {available}"
        )
    
    return PROMPT_TEMPLATES[template_name]
