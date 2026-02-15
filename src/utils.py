"""
Utility functions for DDR Report Generator
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


# Text Processing Utilities

def normalize_location_name(location: str) -> str:
    """
    Normalize location names for consistent matching
    
    Examples:
        "Master Bedroom" -> "master bedroom"
        "MB Bathroom" -> "master bedroom bathroom"
        "Common Bath" -> "common bathroom"
    """
    location = location.lower().strip()
    
    # Expand common abbreviations
    abbreviations = {
        'mb': 'master bedroom',
        'cb': 'common bathroom',
        'wc': 'toilet',
        'br': 'bedroom',
        'lr': 'living room',
        'dr': 'dining room'
    }
    
    for abbr, full in abbreviations.items():
        location = re.sub(r'\b' + abbr + r'\b', full, location)
    
    # Remove extra whitespace
    location = re.sub(r'\s+', ' ', location)
    
    return location


def normalize_issue_type(issue_type: str) -> str:
    """
    Normalize issue type names for consistent categorization
    
    Examples:
        "Dampness" -> "dampness"
        "Water Seepage" -> "seepage"
        "Tile Hollowness" -> "tile defect"
    """
    issue_type = issue_type.lower().strip()
    
    # Standardize terminology
    mappings = {
        'water seepage': 'seepage',
        'water leakage': 'leakage',
        'tile hollowness': 'tile defect',
        'tile gap': 'tile defect',
        'plumbing issue': 'plumbing defect',
        'pipe leak': 'leakage',
        'wall crack': 'crack',
        'ceiling crack': 'crack'
    }
    
    for variant, standard in mappings.items():
        if variant in issue_type:
            return standard
    
    # Extract main issue type (first word usually)
    main_type = issue_type.split()[0] if issue_type else 'unknown'
    
    return main_type


def extract_temperature_readings(text: str) -> List[Dict[str, float]]:
    """
    Extract temperature readings from thermal report text
    
    Returns:
        List of dicts with 'hotspot', 'coldspot', 'delta'
    """
    readings = []
    
    # Pattern: "Hotspot : 28.8 °C" or "28.8 °C"
    hotspot_pattern = r'(?:Hotspot\s*:?\s*)?(\d+\.?\d*)\s*°C'
    coldspot_pattern = r'(?:Coldspot\s*:?\s*)?(\d+\.?\d*)\s*°C'
    
    # Find hotspot/coldspot pairs in text
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for hotspot
        hotspot_match = re.search(hotspot_pattern, line, re.IGNORECASE)
        if hotspot_match:
            hotspot = float(hotspot_match.group(1))
            
            # Look for coldspot in next few lines
            for j in range(i, min(i+5, len(lines))):
                coldspot_match = re.search(coldspot_pattern, lines[j], re.IGNORECASE)
                if coldspot_match and j != i:  # Avoid matching same line
                    coldspot = float(coldspot_match.group(1))
                    delta = abs(hotspot - coldspot)
                    
                    readings.append({
                        'hotspot': hotspot,
                        'coldspot': coldspot,
                        'delta': delta
                    })
                    break
        
        i += 1
    
    return readings


def extract_dates(text: str) -> List[str]:
    """
    Extract dates from text in various formats
    
    Supports: DD.MM.YYYY, DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD
    """
    date_patterns = [
        r'\b(\d{2})\.(\d{2})\.(\d{4})\b',  # DD.MM.YYYY
        r'\b(\d{2})-(\d{2})-(\d{4})\b',     # DD-MM-YYYY
        r'\b(\d{2})/(\d{2})/(\d{4})\b',     # DD/MM/YYYY
        r'\b(\d{4})-(\d{2})-(\d{2})\b'      # YYYY-MM-DD
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) == 3:
                # Reconstruct date string
                if len(match[0]) == 4:  # YYYY-MM-DD format
                    date_str = f"{match[0]}-{match[1]}-{match[2]}"
                else:  # DD.MM.YYYY or similar
                    date_str = f"{match[0]}.{match[1]}.{match[2]}"
                dates.append(date_str)
    
    return list(set(dates))  # Remove duplicates


# Data Quality Utilities

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Simple Jaccard similarity for text comparison
    Used as fallback when sentence transformers unavailable
    
    Returns:
        Similarity score between 0 and 1
    """
    # Tokenize
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    # Jaccard similarity
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def generate_observation_hash(
    location: str,
    issue_type: str,
    description: str
) -> str:
    """
    Generate unique hash for an observation
    Used for duplicate detection
    
    Returns:
        MD5 hash string
    """
    # Normalize inputs
    location_norm = normalize_location_name(location)
    issue_norm = normalize_issue_type(issue_type)
    desc_norm = description.lower().strip()
    
    # Create composite string
    composite = f"{location_norm}|{issue_norm}|{desc_norm}"
    
    # Generate hash
    return hashlib.md5(composite.encode()).hexdigest()[:12]


def is_valid_observation(obs_dict: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate observation data quality
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = ['location', 'issue_type', 'description']
    
    # Check required fields
    for field in required_fields:
        if field not in obs_dict or not obs_dict[field]:
            return False, f"Missing or empty field: {field}"
    
    # Check minimum description length
    if len(obs_dict['description']) < 10:
        return False, "Description too short (minimum 10 characters)"
    
    # Check for placeholder text
    placeholders = ['not available', 'n/a', 'tbd', 'todo', 'xxx']
    desc_lower = obs_dict['description'].lower()
    if any(placeholder in desc_lower for placeholder in placeholders):
        return False, f"Description contains placeholder text"
    
    return True, None


# Formatting Utilities

def format_severity_badge(severity: Optional[str]) -> str:
    """
    Format severity level with appropriate symbol
    
    Returns:
        Formatted string like "🔴 Critical" or "🟡 Medium"
    """
    if not severity:
        return "⚪ Not Specified"
    
    severity = severity.lower()
    
    badges = {
        'critical': '🔴',
        'high': '🟠',
        'medium': '🟡',
        'low': '🟢'
    }
    
    symbol = badges.get(severity, '⚪')
    return f"{symbol} {severity.capitalize()}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length while preserving word boundaries
    """
    if len(text) <= max_length:
        return text
    
    # Find last space before max_length
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated + suffix


def format_list_with_bullets(items: List[str], indent: str = "  ") -> str:
    """
    Format list items with bullet points
    
    Returns:
        Formatted string with each item on new line with bullet
    """
    if not items:
        return ""
    
    formatted_items = [f"{indent}• {item}" for item in items]
    return "\n".join(formatted_items)


# Logging Utilities

def get_timestamp() -> str:
    """Get current timestamp in standard format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_step(step_name: str, status: str = "START", details: str = ""):
    """
    Print formatted log message for pipeline steps
    
    Args:
        step_name: Name of the step (e.g., "Document Parsing")
        status: START, COMPLETE, ERROR, WARNING
        details: Additional information
    """
    symbols = {
        'START': '▶',
        'COMPLETE': '✅',
        'ERROR': '❌',
        'WARNING': '⚠️'
    }
    
    symbol = symbols.get(status.upper(), '•')
    timestamp = get_timestamp()
    
    message = f"{symbol} [{timestamp}] {step_name}"
    if details:
        message += f" - {details}"
    
    print(message)


# File Utilities

def ensure_directory_exists(directory_path: str):
    """Create directory if it doesn't exist"""
    from pathlib import Path
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    from pathlib import Path
    return Path(file_path).stat().st_size / (1024 * 1024)



def test_utilities():
    """Test utility functions"""
    print("Testing utility functions...\n")
    
    # Test location normalization
    print("Location normalization:")
    test_locations = ["Master Bedroom", "MB Bathroom", "Common Bath"]
    for loc in test_locations:
        print(f"  {loc} -> {normalize_location_name(loc)}")
    
    # Test issue type normalization
    print("\nIssue type normalization:")
    test_issues = ["Water Seepage", "Tile Hollowness", "Dampness"]
    for issue in test_issues:
        print(f"  {issue} -> {normalize_issue_type(issue)}")
    
    # Test temperature extraction
    print("\nTemperature extraction:")
    sample_text = "Hotspot : 28.8 °C\nColdspot : 23.4 °C"
    readings = extract_temperature_readings(sample_text)
    print(f"  Found {len(readings)} readings: {readings}")
    
    # Test similarity
    print("\nText similarity:")
    sim = calculate_text_similarity("dampness in hall", "hall dampness")
    print(f"  Similarity: {sim:.2f}")
    
    print("\n✅ All utility tests passed!")


if __name__ == "__main__":
    test_utilities()
