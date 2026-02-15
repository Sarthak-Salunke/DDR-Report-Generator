"""
LLM Provider Abstraction Layer
Supports multiple LLM providers with automatic failover and retry logic
"""

import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
from datetime import datetime
from dataclasses import dataclass, field
import re

# Provider-specific imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Retry logic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)


# Data Models

@dataclass
class UsageStats:
    """Track usage statistics for a provider"""
    provider_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    average_latency_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    
    def update(
        self,
        success: bool,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency_ms: float
    ):
        """Update statistics with new request data"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.total_latency_ms += latency_ms
        self.average_latency_ms = self.total_latency_ms / self.total_requests
        self.last_request_time = datetime.now()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'provider_name': self.provider_name,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': f"{self.success_rate:.2f}%",
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_cost_usd': f"${self.total_cost_usd:.4f}",
            'average_latency_ms': f"{self.average_latency_ms:.2f}ms",
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None
        }


# Custom Exceptions

class LLMProviderError(Exception):
    """Base exception for LLM provider errors"""
    pass


class RateLimitError(LLMProviderError):
    """Rate limit exceeded"""
    pass


class InvalidAPIKeyError(LLMProviderError):
    """Invalid API key"""
    pass


class TimeoutError(LLMProviderError):
    """Request timeout"""
    pass


class ContentPolicyError(LLMProviderError):
    """Content policy violation"""
    pass


# Base Provider Class

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers
    All providers must implement these methods
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 8000,
        temperature: float = 0.1,
        max_retries: int = 3,
        timeout: int = 60,
        verbose: bool = False
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.verbose = verbose
        
        # Usage tracking
        self.stats = UsageStats(provider_name=self.__class__.__name__)
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text response from LLM
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response
            
        Raises:
            LLMProviderError: If generation fails
        """
        pass
    
    @abstractmethod
    def generate_json(self, prompt: str, **kwargs) -> dict:
        """
        Generate JSON response from LLM
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Parsed JSON response as dictionary
            
        Raises:
            LLMProviderError: If generation or parsing fails
        """
        pass
    
    @abstractmethod
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text (provider-specific)
        
        Args:
            text: Input text
            
        Returns:
            Approximate token count
        """
        pass
    
    @abstractmethod
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for request
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        pass
    
    def get_usage_stats(self) -> Dict:
        """Get usage statistics"""
        return self.stats.to_dict()
    
    def _extract_json_from_response(self, response_text: str) -> dict:
        """
        Extract JSON from various LLM response formats
        Handles markdown code blocks, plain JSON, and JSON with preamble
        """
        # Remove markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        else:
            # Try to find JSON object
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                # Find matching closing brace
                brace_count = 0
                for i, char in enumerate(response_text[start:], start=start):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = response_text[start:i+1]
                            break
                else:
                    json_str = response_text.strip()
            else:
                json_str = response_text.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {str(e)}")
            self.logger.debug(f"Attempted to parse: {json_str[:200]}...")
            raise LLMProviderError(f"Invalid JSON response: {str(e)}")


# Gemini Provider

class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini 1.5 Flash provider
    Free tier with generous limits
    """
    
    # Pricing (as of 2024)
    INPUT_COST_PER_1M = 0.075  # $0.075 per 1M input tokens
    OUTPUT_COST_PER_1M = 0.30  # $0.30 per 1M output tokens
    
    def __init__(self, api_key: str, **kwargs):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        
        super().__init__(api_key=api_key, **kwargs)
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize model
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]
        
        self.client = genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        if self.verbose:
            self.logger.info(f"✓ Gemini provider initialized: {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, TimeoutError)),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING)
    )
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response using Gemini"""
        start_time = time.time()
        
        try:
            # Count input tokens
            input_tokens = self._count_tokens(prompt)
            
            # Generate response
            response = self.client.generate_content(prompt)
            
            # Extract text
            if not response.text:
                raise LLMProviderError("Empty response from Gemini")
            
            response_text = response.text
            
            # Count output tokens
            output_tokens = self._count_tokens(response_text)
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            # Update stats
            self.stats.update(
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency_ms=latency_ms
            )
            
            if self.verbose:
                self.logger.info(f"✓ Gemini response: {output_tokens} tokens, {latency_ms:.0f}ms, ${cost:.4f}")
            
            return response_text
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.stats.update(
                success=False,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                latency_ms=latency_ms
            )
            
            # Classify error
            error_msg = str(e).lower()
            if "quota" in error_msg or "rate" in error_msg:
                raise RateLimitError(f"Gemini rate limit: {str(e)}")
            elif "api key" in error_msg or "authentication" in error_msg:
                raise InvalidAPIKeyError(f"Invalid Gemini API key: {str(e)}")
            elif "timeout" in error_msg:
                raise TimeoutError(f"Gemini timeout: {str(e)}")
            else:
                raise LLMProviderError(f"Gemini error: {str(e)}")
    
    def generate_json(self, prompt: str, **kwargs) -> dict:
        """Generate JSON response using Gemini"""
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nIMPORTANT: Respond ONLY with valid JSON. No explanations, no markdown."
        
        response_text = self.generate(json_prompt, **kwargs)
        return self._extract_json_from_response(response_text)
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Gemini request"""
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
        return input_cost + output_cost


# Groq Provider

class GroqProvider(BaseLLMProvider):
    """
    Groq Llama 3.1 70B provider
    Free tier with very fast inference
    """
    
    # Groq free tier (costs may apply in future)
    INPUT_COST_PER_1M = 0.0  # Currently free
    OUTPUT_COST_PER_1M = 0.0  # Currently free
    
    # Rate limits (free tier)
    REQUESTS_PER_MINUTE = 30
    TOKENS_PER_MINUTE = 14400
    
    def __init__(self, api_key: str, **kwargs):
        if not GROQ_AVAILABLE:
            raise ImportError("groq package not installed. Run: pip install groq")
        
        super().__init__(api_key=api_key, **kwargs)
        
        # Initialize Groq client
        self.client = Groq(api_key=api_key)
        
        if self.verbose:
            self.logger.info(f"✓ Groq provider initialized: {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, TimeoutError)),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING)
    )
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response using Groq"""
        start_time = time.time()
        
        try:
            # Count input tokens
            input_tokens = self._count_tokens(prompt)
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
            
            # Extract text
            if not response.choices or not response.choices[0].message.content:
                raise LLMProviderError("Empty response from Groq")
            
            response_text = response.choices[0].message.content
            
            # Get actual token counts from response
            if hasattr(response, 'usage'):
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
            else:
                output_tokens = self._count_tokens(response_text)
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            # Update stats
            self.stats.update(
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency_ms=latency_ms
            )
            
            if self.verbose:
                self.logger.info(f"✓ Groq response: {output_tokens} tokens, {latency_ms:.0f}ms, ${cost:.4f}")
            
            return response_text
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.stats.update(
                success=False,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                latency_ms=latency_ms
            )
            
            # Classify error
            error_msg = str(e).lower()
            if "rate_limit" in error_msg or "429" in error_msg:
                raise RateLimitError(f"Groq rate limit: {str(e)}")
            elif "api_key" in error_msg or "401" in error_msg or "authentication" in error_msg:
                raise InvalidAPIKeyError(f"Invalid Groq API key: {str(e)}")
            elif "timeout" in error_msg:
                raise TimeoutError(f"Groq timeout: {str(e)}")
            else:
                raise LLMProviderError(f"Groq error: {str(e)}")
    
    def generate_json(self, prompt: str, **kwargs) -> dict:
        """Generate JSON response using Groq"""
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nIMPORTANT: Respond ONLY with valid JSON. No explanations, no markdown."
        
        response_text = self.generate(json_prompt, **kwargs)
        return self._extract_json_from_response(response_text)
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count (4 chars ≈ 1 token for Llama)"""
        return len(text) // 4
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Groq request (currently free)"""
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
        return input_cost + output_cost


# Utility Functions

def create_provider(
    provider_name: str,
    api_key: str,
    model: str,
    **kwargs
) -> BaseLLMProvider:
    """
    Factory function to create provider instances
    
    Args:
        provider_name: Name of provider ('gemini', 'groq')
        api_key: API key for provider
        model: Model name
        **kwargs: Additional provider parameters
        
    Returns:
        Initialized provider instance
        
    Raises:
        ValueError: If provider name is invalid
    """
    providers = {
        'gemini': GeminiProvider,
        'groq': GroqProvider,
    }
    
    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
    
    return provider_class(api_key=api_key, model=model, **kwargs)


def print_usage_summary(providers: List[BaseLLMProvider]):
    """
    Print usage summary for multiple providers
    
    Args:
        providers: List of provider instances
    """
    print("\n" + "="*70)
    print("LLM Provider Usage Summary")
    print("="*70)
    
    for provider in providers:
        stats = provider.get_usage_stats()
        print(f"\n{stats['provider_name']}:")
        print(f"  Requests: {stats['total_requests']} ({stats['success_rate']} success)")
        print(f"  Tokens: {stats['total_input_tokens']:,} in / {stats['total_output_tokens']:,} out")
        print(f"  Cost: {stats['total_cost_usd']}")
        print(f"  Avg Latency: {stats['average_latency_ms']}")
    
    print("="*70 + "\n")
