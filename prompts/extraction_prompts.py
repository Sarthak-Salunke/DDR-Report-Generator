"""
Extraction prompts for LLM-based data extraction
All prompts follow strict grounding principles to prevent hallucination
"""

# Inspection Report Extraction Prompt

INSPECTION_EXTRACTION_PROMPT = """You are a professional building inspection data extraction specialist. Your task is to extract structured information from an inspection report with ABSOLUTE accuracy - never invent or assume information not explicitly stated in the document.

## CRITICAL RULES:
1. Extract ONLY information explicitly present in the document
2. If information is unclear or missing, use "Not Available" 
3. Do NOT infer or assume any details
4. Maintain exact terminology from the source document
5. Include specific measurements, dates, and references when provided

## INPUT DOCUMENT:
{document_text}

## EXTRACTION TASK:
Extract the following information and return as valid JSON:

1. **Property Details:**
   - property_type: Type of property (e.g., "Flat", "House")
   - floors: Number of floors (integer or null)
   - inspection_date: Date of inspection (DD.MM.YYYY format)
   - inspected_by: List of inspector names
   - score: Overall inspection score (0-100, or null)
   - flagged_items: Number of flagged items (integer or null)

2. **Observations:**
   For each impacted area/issue, extract:
   - location: Specific area/room name (e.g., "Hall", "Master Bedroom")
   - issue_type: Type of defect (e.g., "Dampness", "Crack", "Seepage")
   - description: Detailed description of the observation
   - severity: If mentioned, one of: "Critical", "High", "Medium", "Low", or null
   - evidence: Photo reference numbers or measurements if mentioned
   - confidence: Your confidence in this extraction (0.0-1.0)

3. **Root Causes:**
   List of identified root causes mentioned in the report

## OUTPUT FORMAT:
Return ONLY valid JSON in this exact structure:
```json
{{
  "property_details": {{
    "property_type": "string",
    "floors": integer or null,
    "inspection_date": "DD.MM.YYYY",
    "inspected_by": ["name1", "name2"],
    "score": float or null,
    "flagged_items": integer or null
  }},
  "observations": [
    {{
      "location": "string",
      "issue_type": "string",
      "description": "string",
      "severity": "string or null",
      "source_document": "inspection",
      "confidence": float,
      "evidence": "string or null"
    }}
  ],
  "root_causes": ["cause1", "cause2"],
  "metadata": {{
    "total_observations": integer,
    "extraction_timestamp": "ISO timestamp"
  }}
}}
```

## EXAMPLE EXTRACTION:
If the document states:
"Observed dampness at the skirting level of Hall of Flat No. 103"

Extract as:
```json
{{
  "location": "Hall",
  "issue_type": "Dampness",
  "description": "Dampness observed at skirting level",
  "severity": null,
  "source_document": "inspection",
  "confidence": 0.9,
  "evidence": "Photo 1-7"
}}
```

Now extract ALL observations from the provided document following these rules strictly.
"""

# Thermal Report Extraction Prompt

THERMAL_EXTRACTION_PROMPT = """You are a thermal imaging data extraction specialist. Extract temperature readings and thermal anomalies from this thermal imaging report with complete accuracy.

## CRITICAL RULES:
1. Extract ONLY explicitly stated temperature values
2. Never calculate or estimate temperatures
3. Include all temperature readings found (hotspot, coldspot, delta)
4. Preserve exact location references from the document
5. Use "Not Available" for any missing information

## INPUT DOCUMENT:
{document_text}

## EXTRACTION TASK:

Extract thermal readings with the following information:

1. **Thermal Readings:**
   For each thermal image/reading:
   - location: Area where reading taken (as stated in document)
   - hotspot_temp: Maximum temperature in °C (float)
   - coldspot_temp: Minimum temperature in °C (float)
   - temperature_delta: Temperature difference in °C (float)
   - date: Date of thermal imaging (DD.MM.YYYY)
   - image_reference: Thermal image ID/filename (e.g., "RB02380X.JPG")
   - interpretation: Brief description of what the thermal anomaly indicates

2. **Device Information:**
   - device: Thermal camera model if mentioned
   - date: Overall session date

## OUTPUT FORMAT:
Return ONLY valid JSON:
```json
{{
  "readings": [
    {{
      "location": "string",
      "hotspot_temp": float,
      "coldspot_temp": float,
      "temperature_delta": float,
      "date": "DD.MM.YYYY",
      "image_reference": "string",
      "interpretation": "string"
    }}
  ],
  "date": "DD.MM.YYYY",
  "device": "string or null",
  "metadata": {{
    "total_readings": integer,
    "extraction_timestamp": "ISO timestamp"
  }}
}}
```

## EXAMPLE EXTRACTION:
If document shows:
"Thermal image : RB02380X.JPG
Hotspot : 28.8 °C
Coldspot : 23.4 °C
27/09/22"

Extract as:
```json
{{
  "location": "Not Available",
  "hotspot_temp": 28.8,
  "coldspot_temp": 23.4,
  "temperature_delta": 5.4,
  "date": "27.09.2022",
  "image_reference": "RB02380X.JPG",
  "interpretation": "Temperature differential of 5.4°C indicates potential moisture or thermal bridge"
}}
```

Now extract ALL thermal readings from the document.
"""


VALIDATION_PROMPT = """You are a quality assurance specialist. Review the extracted data and validate its accuracy against the source document.

## SOURCE DOCUMENT:
{source_document}

## EXTRACTED DATA:
{extracted_data}

## VALIDATION TASK:
Check for:
1. **Accuracy**: All extracted information present in source document
2. **Completeness**: No major information missed
3. **Hallucinations**: No invented or assumed information
4. **Format**: Proper data types and structure

Return validation report as JSON:
```json
{{
  "is_valid": boolean,
  "accuracy_score": float (0-1),
  "errors_found": [
    {{
      "type": "hallucination|missing|incorrect",
      "field": "field_name",
      "description": "description of error"
    }}
  ],
  "warnings": ["list of minor issues"],
  "summary": "brief summary of validation results"
}}
```
"""

# Conflict Resolution Prompt

CONFLICT_RESOLUTION_PROMPT = """You are a data conflict resolution specialist. Two sources provide information about the same observation but with different details.

## OBSERVATION 1 (Source: {source1}):
{observation1}

## OBSERVATION 2 (Source: {source2}):
{observation2}

## RESOLUTION TASK:
1. Identify if these refer to the same issue in the same location
2. Determine if there is a true conflict or just different perspectives
3. If conflict exists, flag it explicitly
4. Suggest resolution strategy

Return as JSON:
```json
{{
  "same_issue": boolean,
  "conflict_exists": boolean,
  "conflict_type": "measurement|description|severity|timing",
  "merged_observation": {{
    "location": "string",
    "issue_type": "string",
    "description": "merged description",
    "sources": ["source1", "source2"],
    "conflict_note": "description of conflict if any"
  }},
  "confidence": float (0-1),
  "recommended_action": "string"
}}
```
"""


def build_inspection_prompt(document_text: str) -> str:
    """Build complete prompt for inspection data extraction"""
    return INSPECTION_EXTRACTION_PROMPT.format(document_text=document_text)


def build_thermal_prompt(document_text: str) -> str:
    """Build complete prompt for thermal data extraction"""
    return THERMAL_EXTRACTION_PROMPT.format(document_text=document_text)


def build_validation_prompt(source_document: str, extracted_data: str) -> str:
    """Build prompt for validating extracted data"""
    return VALIDATION_PROMPT.format(
        source_document=source_document,
        extracted_data=extracted_data
    )


def build_conflict_resolution_prompt(
    observation1: dict,
    observation2: dict,
    source1: str,
    source2: str
) -> str:
    """Build prompt for resolving conflicts between observations"""
    import json
    return CONFLICT_RESOLUTION_PROMPT.format(
        source1=source1,
        source2=source2,
        observation1=json.dumps(observation1, indent=2),
        observation2=json.dumps(observation2, indent=2)
    )
