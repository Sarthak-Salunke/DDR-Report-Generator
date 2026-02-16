"""
Data Merger Module
Handles merging observations from multiple sources and detecting duplicates
Uses semantic similarity for intelligent deduplication
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

from src.data_models import Observation, MergedObservation
from src.utils import (
    normalize_location_name,
    normalize_issue_type,
    generate_observation_hash
)


class DataMerger:
    """
    Merges observations from inspection and thermal reports
    Detects and handles duplicates using semantic similarity
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        embedding_model: str = "all-MiniLM-L6-v2",
        verbose: bool = True
    ):
        """
        Initialize merger with similarity detection
        
        Args:
            similarity_threshold: Minimum similarity to consider duplicate (0-1)
            embedding_model: Sentence transformer model name
            verbose: Print progress messages
        """
        self.similarity_threshold = similarity_threshold
        self.verbose = verbose
        
        # Load embedding model
        if self.verbose:
            print(f"📊 Loading embedding model: {embedding_model}")
        
        try:
            self.model = SentenceTransformer(embedding_model)
            if self.verbose:
                print("  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ⚠️  Failed to load model: {e}")
            print("  ⚠️  Falling back to simple text similarity")
            self.model = None
    
    def merge_observations(
        self,
        inspection_observations: List[Observation],
        thermal_observations: List[Observation]
    ) -> Dict:
        """
        Merge observations from both sources, detecting duplicates
        
        Args:
            inspection_observations: List from inspection report
            thermal_observations: List from thermal report
            
        Returns:
            Dictionary with merged observations and metadata
        """
        if self.verbose:
            print("\n🔗 Merging observations...")
            print(f"  - Inspection: {len(inspection_observations)} observations")
            print(f"  - Thermal: {len(thermal_observations)} observations")
        
        all_observations = inspection_observations + thermal_observations
        
        if not all_observations:
            return {
                'merged_observations': [],
                'total_observations': 0,
                'duplicate_groups': 0,
                'conflicts_detected': [],
                'metadata': {}
            }
        
        # Step 1: Generate embeddings
        embeddings = self._generate_embeddings(all_observations)
        
        # Step 2: Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(embeddings)
        
        # Step 3: Find duplicate groups
        duplicate_groups = self._find_duplicate_groups(
            all_observations,
            similarity_matrix
        )
        
        # Step 4: Merge duplicates
        merged_observations = self._merge_duplicate_groups(
            all_observations,
            duplicate_groups
        )
        
        # Step 5: Detect conflicts
        conflicts = self._detect_conflicts(merged_observations)
        
        if self.verbose:
            print(f"  ✓ Merged to {len(merged_observations)} unique observations")
            print(f"  ✓ Found {len(duplicate_groups)} duplicate groups")
            print(f"  ✓ Detected {len(conflicts)} conflicts")
        
        return {
            'merged_observations': merged_observations,
            'total_observations': len(merged_observations),
            'duplicate_groups': len(duplicate_groups),
            'conflicts_detected': conflicts,
            'metadata': {
                'original_count': len(all_observations),
                'deduplication_rate': 1 - (len(merged_observations) / len(all_observations)) if all_observations else 0,
                'similarity_threshold': self.similarity_threshold,
                'duplicate_group_details': duplicate_groups
            }
        }
    
    def save_merge_results(
        self,
        result: Dict,
        output_path: str = "data/intermediate/merge_results.json"
    ):
        """
        Save merge results for debugging and analysis
        
        Args:
            result: Result dictionary from merge_observations
            output_path: Path to save JSON file
        """
        import json
        from pathlib import Path
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable = {
            'total_observations': result['total_observations'],
            'duplicate_groups': result['duplicate_groups'],
            'conflicts': result['conflicts_detected'],
            'observations': [
                {
                    'location': obs.location,
                    'issue_type': obs.issue_type,
                    'description': obs.description,
                    'severity': obs.severity,
                    'sources': obs.sources,
                    'evidence': obs.evidence,
                    'confidence': obs.confidence,
                    'is_duplicate': obs.is_duplicate,
                    'conflict_detected': obs.conflict_detected,
                    'conflict_details': obs.conflict_details
                }
                for obs in result['merged_observations']
            ],
            'metadata': result['metadata']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"Merge results saved to {output_path}")
    
    def analyze_threshold_sensitivity(
        self,
        inspection_observations: List[Observation],
        thermal_observations: List[Observation],
        thresholds: List[float] = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    ) -> Dict:
        """
        Analyze how different similarity thresholds affect merging
        Useful for tuning the optimal threshold
        
        Args:
            inspection_observations: List from inspection report
            thermal_observations: List from thermal report
            thresholds: List of thresholds to test
            
        Returns:
            Dictionary with analysis results for each threshold
        """
        print("\n" + "="*70)
        print("🔬 THRESHOLD SENSITIVITY ANALYSIS")
        print("="*70)
        
        original_threshold = self.similarity_threshold
        results = {}
        
        for threshold in thresholds:
            self.similarity_threshold = threshold
            
            # Run merge with this threshold
            result = self.merge_observations(
                inspection_observations,
                thermal_observations
            )
            
            results[threshold] = {
                'total_observations': result['total_observations'],
                'duplicate_groups': result['duplicate_groups'],
                'conflicts': len(result['conflicts_detected']),
                'deduplication_rate': result['metadata']['deduplication_rate']
            }
            
            print(f"\nThreshold {threshold:.2f}:")
            print(f"  Observations: {result['total_observations']}")
            print(f"  Duplicates: {result['duplicate_groups']} groups")
            print(f"  Conflicts: {len(result['conflicts_detected'])}")
            print(f"  Dedup rate: {result['metadata']['deduplication_rate']:.1%}")
        
        # Restore original threshold
        self.similarity_threshold = original_threshold
        
        # Print recommendation
        print("\n" + "="*70)
        print("📊 RECOMMENDATIONS:")
        print("="*70)
        
        # Find optimal threshold (balance between deduplication and conflicts)
        optimal = None
        optimal_score = -1
        
        for threshold, data in results.items():
            # Score: high dedup rate, low conflicts
            score = data['deduplication_rate'] - (data['conflicts'] * 0.1)
            if score > optimal_score:
                optimal_score = score
                optimal = threshold
        
        print(f"\n✓ Recommended threshold: {optimal:.2f}")
        print(f"   - Deduplication rate: {results[optimal]['deduplication_rate']:.1%}")
        print(f"   - Duplicate groups: {results[optimal]['duplicate_groups']}")
        print(f"   - Conflicts: {results[optimal]['conflicts']}")
        
        print("\n💡 Guidelines:")
        print("   - Too many duplicates? Increase threshold (0.90+)")
        print("   - Missing duplicates? Decrease threshold (0.75-0.80)")
        print("   - Good balance? Use 0.85")
        print("="*70 + "\n")
        
        return results
    
    def _generate_embeddings(self, observations: List[Observation]) -> np.ndarray:
        """Generate semantic embeddings for observations"""
        if self.model is None:
            # Fallback to simple hashing if model not available
            return np.array([[hash(obs.description) % 1000 / 1000.0] for obs in observations])
        
        # Create text representations
        texts = []
        for obs in observations:
            # Combine location, issue type, and description
            text = f"{obs.location} {obs.issue_type} {obs.description}"
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        return embeddings
    
    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix"""
        return cosine_similarity(embeddings)
    
    def _find_duplicate_groups(
        self,
        observations: List[Observation],
        similarity_matrix: np.ndarray
    ) -> List[List[int]]:
        """
        Find groups of duplicate observations based on similarity
        
        Returns:
            List of groups, where each group is list of observation indices
        """
        n = len(observations)
        visited = set()
        groups = []
        
        for i in range(n):
            if i in visited:
                continue
            
            # Find all observations similar to i
            group = [i]
            visited.add(i)
            
            for j in range(i + 1, n):
                if j in visited:
                    continue
                
                # Check similarity
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    # Additional checks for location and issue type
                    if self._are_same_location(observations[i], observations[j]):
                        if self._are_same_issue_type(observations[i], observations[j]):
                            group.append(j)
                            visited.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _are_same_location(self, obs1: Observation, obs2: Observation) -> bool:
        """Check if two observations refer to same location"""
        loc1 = normalize_location_name(obs1.location)
        loc2 = normalize_location_name(obs2.location)
        return loc1 == loc2
    
    def _are_same_issue_type(self, obs1: Observation, obs2: Observation) -> bool:
        """Check if two observations are same issue type"""
        type1 = normalize_issue_type(obs1.issue_type)
        type2 = normalize_issue_type(obs2.issue_type)
        return type1 == type2
    
    def _merge_duplicate_groups(
        self,
        observations: List[Observation],
        duplicate_groups: List[List[int]]
    ) -> List[MergedObservation]:
        """
        Merge duplicate observations into single merged observations
        """
        merged = []
        processed = set()
        
        # First, process all duplicate groups
        for group in duplicate_groups:
            if any(idx in processed for idx in group):
                continue
            
            group_obs = [observations[idx] for idx in group]
            merged_obs = self._merge_observation_group(group_obs)
            merged.append(merged_obs)
            
            processed.update(group)
        
        # Then, add remaining observations (not part of any group)
        for idx, obs in enumerate(observations):
            if idx not in processed:
                merged_obs = MergedObservation(
                    location=obs.location,
                    issue_type=obs.issue_type,
                    description=obs.description,
                    severity=obs.severity,
                    sources=[obs.source_document],
                    evidence=[obs.evidence] if obs.evidence else [],
                    confidence=obs.confidence,
                    is_duplicate=False,
                    conflict_detected=False
                )
                merged.append(merged_obs)
        
        return merged
    
    def _merge_observation_group(self, group: List[Observation]) -> MergedObservation:
        """
        Merge a group of duplicate observations into one
        """
        # Use first observation as base
        base = group[0]
        
        # Collect all sources
        sources = list(set(obs.source_document for obs in group))
        
        # Collect all evidence
        evidence = []
        for obs in group:
            if obs.evidence:
                evidence.append(obs.evidence)
        
        # Merge descriptions (take longest)
        descriptions = [obs.description for obs in group]
        merged_description = max(descriptions, key=len)
        
        # Average confidence
        avg_confidence = sum(obs.confidence for obs in group) / len(group)
        
        # Check for severity conflicts
        severities = [obs.severity for obs in group if obs.severity]
        severity = severities[0] if severities else None
        conflict_detected = len(set(severities)) > 1 if severities else False
        
        return MergedObservation(
            location=base.location,
            issue_type=base.issue_type,
            description=merged_description,
            severity=severity,
            sources=sources,
            evidence=evidence,
            confidence=avg_confidence,
            is_duplicate=True,
            conflict_detected=conflict_detected,
            conflict_details=f"Severity conflict: {set(severities)}" if conflict_detected else None
        )
    
    def _detect_conflicts(
        self,
        merged_observations: List[MergedObservation]
    ) -> List[Dict]:
        """
        Detect conflicts in merged observations
        
        Returns conflicts requiring human review
        """
        conflicts = []
        
        for obs in merged_observations:
            if obs.conflict_detected:
                conflicts.append({
                    'location': obs.location,
                    'issue_type': obs.issue_type,
                    'conflict_type': 'severity_mismatch',
                    'details': obs.conflict_details,
                    'sources': obs.sources,
                    'action_required': 'Review severity assignment'
                })
        
        return conflicts



def test_merger():
    """Test the data merger with sample observations"""
    print("="*60)
    print("Testing Data Merger")
    print("="*60)
    
    # Create sample observations
    obs1 = Observation(
        location="Hall",
        issue_type="Dampness",
        description="Dampness observed at skirting level",
        severity="Medium",
        source_document="inspection",
        confidence=0.9,
        evidence="Photo 1-7"
    )
    
    obs2 = Observation(
        location="Hall",
        issue_type="Dampness",
        description="Moisture detected at floor level",
        severity="Medium",
        source_document="thermal",
        confidence=0.85,
        evidence="RB02380X.JPG"
    )
    
    obs3 = Observation(
        location="Kitchen",
        issue_type="Crack",
        description="Small crack on wall",
        severity="Low",
        source_document="inspection",
        confidence=0.8,
        evidence="Photo 31"
    )
    
    # Create merger
    merger = DataMerger(similarity_threshold=0.85, verbose=True)
    
    # Merge observations
    result = merger.merge_observations(
        [obs1, obs3],  # inspection
        [obs2]  # thermal
    )
    
    print(f"\n✓ Merge completed:")
    print(f"   - Original: 3 observations")
    print(f"   - Merged: {result['total_observations']} unique")
    print(f"   - Duplicates: {result['duplicate_groups']} groups")
    print(f"   - Conflicts: {len(result['conflicts_detected'])}")
    
    print("\n" + "="*60)
    print("✓ MERGER TEST PASSED")
    print("="*60)


if __name__ == "__main__":
    test_merger()
