"""
Comprehensive tests for LLM providers and manager
Tests provider functionality, failover, and error handling
"""

import os
import pytest
from dotenv import load_dotenv

from src.llm_providers import (
    GeminiProvider,
    GroqProvider,
    create_provider,
    LLMProviderError,
    RateLimitError,
    InvalidAPIKeyError
)
from src.llm_manager import LLMManager


# Load environment variables
load_dotenv()


# Provider Tests

class TestGeminiProvider:
    """Test Gemini provider functionality"""
    
    @pytest.fixture
    def provider(self):
        """Create Gemini provider instance"""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not set")
        
        return GeminiProvider(
            api_key=api_key,
            model="gemini-1.5-flash",
            max_tokens=1000,
            temperature=0.1,
            verbose=True
        )
    
    def test_simple_generation(self, provider):
        """Test simple text generation"""
        prompt = "What is 2+2? Answer in one sentence."
        response = provider.generate(prompt)
        
        assert response is not None
        assert len(response) > 0
        assert "4" in response or "four" in response.lower()
    
    def test_json_generation(self, provider):
        """Test JSON generation"""
        prompt = """
        Extract the following as JSON:
        - name: John Doe
        - age: 30
        - city: New York
        """
        
        response = provider.generate_json(prompt)
        
        assert isinstance(response, dict)
        assert 'name' in response or 'Name' in response
    
    def test_usage_stats(self, provider):
        """Test usage statistics tracking"""
        prompt = "Hello, world!"
        provider.generate(prompt)
        
        stats = provider.get_usage_stats()
        
        assert stats['total_requests'] >= 1
        assert stats['successful_requests'] >= 1
        assert float(stats['total_cost_usd'].replace('$', '')) >= 0
    
    def test_invalid_api_key(self):
        """Test handling of invalid API key"""
        provider = GeminiProvider(
            api_key="invalid_key_12345",
            model="gemini-1.5-flash",
            verbose=False
        )
        
        with pytest.raises((InvalidAPIKeyError, LLMProviderError)):
            provider.generate("Test prompt")


class TestGroqProvider:
    """Test Groq provider functionality"""
    
    @pytest.fixture
    def provider(self):
        """Create Groq provider instance"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            pytest.skip("GROQ_API_KEY not set")
        
        return GroqProvider(
            api_key=api_key,
            model="llama-3.1-70b-versatile",
            max_tokens=1000,
            temperature=0.1,
            verbose=True
        )
    
    def test_simple_generation(self, provider):
        """Test simple text generation"""
        prompt = "What is the capital of France? Answer in one word."
        response = provider.generate(prompt)
        
        assert response is not None
        assert len(response) > 0
        assert "Paris" in response
    
    def test_json_generation(self, provider):
        """Test JSON generation"""
        prompt = """
        Create a JSON object with:
        - color: blue
        - number: 42
        - active: true
        """
        
        response = provider.generate_json(prompt)
        
        assert isinstance(response, dict)
        assert len(response) > 0
    
    def test_usage_stats(self, provider):
        """Test usage statistics tracking"""
        prompt = "Hello!"
        provider.generate(prompt)
        
        stats = provider.get_usage_stats()
        
        assert stats['total_requests'] >= 1
        assert stats['successful_requests'] >= 1


class TestProviderFactory:
    """Test provider factory function"""
    
    def test_create_gemini_provider(self):
        """Test creating Gemini provider"""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not set")
        
        provider = create_provider(
            provider_name='gemini',
            api_key=api_key,
            model='gemini-1.5-flash'
        )
        
        assert isinstance(provider, GeminiProvider)
    
    def test_create_groq_provider(self):
        """Test creating Groq provider"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            pytest.skip("GROQ_API_KEY not set")
        
        provider = create_provider(
            provider_name='groq',
            api_key=api_key,
            model='llama-3.1-70b-versatile'
        )
        
        assert isinstance(provider, GroqProvider)
    
    def test_invalid_provider_name(self):
        """Test handling of invalid provider name"""
        with pytest.raises(ValueError):
            create_provider(
                provider_name='invalid_provider',
                api_key='test_key',
                model='test_model'
            )


# LLM Manager Tests

class TestLLMManager:
    """Test LLM Manager functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return {
            'primary_provider': 'gemini',
            'backup_provider': 'groq',
            'failure_threshold': 3,
            'circuit_breaker_minutes': 5,
            'auto_recovery_minutes': 30,
            'gemini': {
                'model': 'gemini-1.5-flash',
                'api_key_env': 'GOOGLE_API_KEY',
                'max_tokens': 1000,
                'temperature': 0.1,
                'max_retries': 3,
                'timeout': 60
            },
            'groq': {
                'model': 'llama-3.1-70b-versatile',
                'api_key_env': 'GROQ_API_KEY',
                'max_tokens': 1000,
                'temperature': 0.1,
                'max_retries': 3,
                'timeout': 60
            }
        }
    
    @pytest.fixture
    def manager(self, config):
        """Create LLM Manager instance"""
        # Check if API keys are available
        if not os.getenv('GOOGLE_API_KEY') and not os.getenv('GROQ_API_KEY'):
            pytest.skip("No API keys available")
        
        return LLMManager(config=config, verbose=True)
    
    def test_simple_generation(self, manager):
        """Test simple text generation with manager"""
        prompt = "What is 5+5? Answer in one sentence."
        response = manager.generate(prompt, task="math_test")
        
        assert response is not None
        assert len(response) > 0
        assert "10" in response or "ten" in response.lower()
    
    def test_json_generation(self, manager):
        """Test JSON generation with manager"""
        prompt = """
        Create a JSON object representing a person:
        - name: Alice
        - age: 25
        - occupation: Engineer
        """
        
        response = manager.generate_json(prompt, task="json_test")
        
        assert isinstance(response, dict)
        assert len(response) > 0
    
    def test_health_status(self, manager):
        """Test health status tracking"""
        # Make a request
        manager.generate("Hello!", task="health_test")
        
        health = manager.get_health_status()
        
        assert 'providers' in health
        assert 'using_backup' in health
        assert isinstance(health['using_backup'], bool)
    
    def test_usage_stats(self, manager):
        """Test usage statistics"""
        # Make a request
        manager.generate("Test", task="stats_test")
        
        stats = manager.get_usage_stats()
        
        assert len(stats) > 0
        for provider_stats in stats.values():
            assert 'total_requests' in provider_stats


# Integration Tests

class TestIntegration:
    """Integration tests for full workflow"""
    
    def test_failover_simulation(self):
        """Test failover behavior (requires both API keys)"""
        if not os.getenv('GOOGLE_API_KEY') or not os.getenv('GROQ_API_KEY'):
            pytest.skip("Both API keys required for failover test")
        
        config = {
            'primary_provider': 'gemini',
            'backup_provider': 'groq',
            'failure_threshold': 1,  # Fail fast for testing
            'circuit_breaker_minutes': 1,
            'gemini': {
                'model': 'gemini-1.5-flash',
                'api_key_env': 'GOOGLE_API_KEY',
                'max_tokens': 100,
                'temperature': 0.1,
                'max_retries': 1,
                'timeout': 5
            },
            'groq': {
                'model': 'llama-3.1-70b-versatile',
                'api_key_env': 'GROQ_API_KEY',
                'max_tokens': 100,
                'temperature': 0.1,
                'max_retries': 1,
                'timeout': 5
            }
        }
        
        manager = LLMManager(config=config, verbose=True)
        
        # Should work with at least one provider
        response = manager.generate("Hello!", task="failover_test")
        assert response is not None
    
    def test_cost_tracking(self):
        """Test cost tracking across providers"""
        if not os.getenv('GOOGLE_API_KEY'):
            pytest.skip("GOOGLE_API_KEY required")
        
        config = {
            'primary_provider': 'gemini',
            'backup_provider': 'groq',
            'gemini': {
                'model': 'gemini-1.5-flash',
                'api_key_env': 'GOOGLE_API_KEY',
                'max_tokens': 100,
                'temperature': 0.1
            }
        }
        
        manager = LLMManager(config=config, verbose=True)
        
        # Make several requests
        for i in range(3):
            manager.generate(f"Test {i}", task=f"cost_test_{i}")
        
        stats = manager.get_usage_stats()
        
        # Check that costs are tracked
        for provider_stats in stats.values():
            if provider_stats['total_requests'] > 0:
                cost_str = provider_stats['total_cost_usd']
                assert cost_str.startswith('$')
                cost = float(cost_str.replace('$', ''))
                assert cost >= 0


# Performance Tests

class TestPerformance:
    """Performance and latency tests"""
    
    def test_response_time(self):
        """Test that responses are reasonably fast"""
        if not os.getenv('GOOGLE_API_KEY'):
            pytest.skip("GOOGLE_API_KEY required")
        
        import time
        
        provider = GeminiProvider(
            api_key=os.getenv('GOOGLE_API_KEY'),
            model='gemini-1.5-flash',
            verbose=False
        )
        
        start = time.time()
        provider.generate("Hello!")
        elapsed = time.time() - start
        
        # Should respond within 10 seconds
        assert elapsed < 10.0
    
    def test_batch_processing(self):
        """Test processing multiple requests"""
        if not os.getenv('GOOGLE_API_KEY'):
            pytest.skip("GOOGLE_API_KEY required")
        
        config = {
            'primary_provider': 'gemini',
            'gemini': {
                'model': 'gemini-1.5-flash',
                'api_key_env': 'GOOGLE_API_KEY',
                'max_tokens': 100,
                'temperature': 0.1
            }
        }
        
        manager = LLMManager(config=config, verbose=False)
        
        # Process multiple requests
        prompts = [f"Count to {i}" for i in range(1, 4)]
        responses = []
        
        for prompt in prompts:
            response = manager.generate(prompt, task="batch_test")
            responses.append(response)
        
        assert len(responses) == len(prompts)
        assert all(r is not None for r in responses)


# Run Tests

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
