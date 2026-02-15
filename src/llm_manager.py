"""
LLM Manager - Orchestrates multiple providers with failover
Handles provider selection, health tracking, and automatic fallback
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from src.llm_providers import (
    BaseLLMProvider,
    create_provider,
    LLMProviderError,
    RateLimitError,
    InvalidAPIKeyError,
    TimeoutError,
    print_usage_summary
)


# Provider Health Tracking

@dataclass
class ProviderHealth:
    """Track health status of a provider"""
    provider_name: str
    is_healthy: bool = True
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_failures: int = 0
    total_successes: int = 0
    circuit_breaker_until: Optional[datetime] = None
    
    def record_success(self):
        """Record successful request"""
        self.is_healthy = True
        self.consecutive_failures = 0
        self.last_success_time = datetime.now()
        self.total_successes += 1
        self.circuit_breaker_until = None
    
    def record_failure(self, failure_threshold: int = 3, circuit_breaker_minutes: int = 5):
        """Record failed request and update health status"""
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now()
        self.total_failures += 1
        
        # Open circuit breaker if threshold exceeded
        if self.consecutive_failures >= failure_threshold:
            self.is_healthy = False
            self.circuit_breaker_until = datetime.now() + timedelta(minutes=circuit_breaker_minutes)
    
    def should_try(self) -> bool:
        """Check if provider should be tried"""
        # If circuit breaker is active, check if it's time to retry
        if self.circuit_breaker_until:
            if datetime.now() >= self.circuit_breaker_until:
                # Reset circuit breaker
                self.circuit_breaker_until = None
                self.is_healthy = True
                return True
            return False
        
        return self.is_healthy
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        total = self.total_successes + self.total_failures
        if total == 0:
            return 100.0
        return (self.total_successes / total) * 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'provider_name': self.provider_name,
            'is_healthy': self.is_healthy,
            'consecutive_failures': self.consecutive_failures,
            'success_rate': f"{self.success_rate:.2f}%",
            'total_successes': self.total_successes,
            'total_failures': self.total_failures,
            'circuit_breaker_active': self.circuit_breaker_until is not None,
            'last_success': self.last_success_time.isoformat() if self.last_success_time else None,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


# LLM Manager

class LLMManager:
    """
    Manages multiple LLM providers with automatic failover
    
    Features:
    - Automatic provider selection based on priority
    - Health tracking and circuit breaker pattern
    - Automatic failover on errors
    - Recovery to primary provider after successful backup usage
    - Comprehensive logging and metrics
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        verbose: bool = True
    ):
        """
        Initialize LLM Manager
        
        Args:
            config: Configuration dictionary with provider settings
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # Initialize providers
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.health: Dict[str, ProviderHealth] = {}
        self.provider_priority: List[str] = []
        
        # Failover settings
        self.failure_threshold = config.get('failure_threshold', 3)
        self.circuit_breaker_minutes = config.get('circuit_breaker_minutes', 5)
        self.auto_recovery_minutes = config.get('auto_recovery_minutes', 30)
        
        # Track last primary success for recovery
        self.last_primary_success: Optional[datetime] = None
        self.using_backup = False
        
        self._initialize_providers()
        
        if self.verbose:
            self.logger.info(f"✓ LLMManager initialized with {len(self.providers)} providers")
            self.logger.info(f"  Priority order: {' → '.join(self.provider_priority)}")
    
    def _initialize_providers(self):
        """Initialize all configured providers"""
        # Get provider priority from config
        primary = self.config.get('primary_provider', 'gemini')
        backup = self.config.get('backup_provider', 'groq')
        
        self.provider_priority = [primary, backup]
        
        # Initialize each provider
        for provider_name in self.provider_priority:
            provider_config = self.config.get(provider_name, {})
            
            if not provider_config:
                self.logger.warning(f"No configuration found for provider: {provider_name}")
                continue
            
            try:
                # Get API key from environment variable
                import os
                api_key_env = provider_config.get('api_key_env')
                api_key = os.getenv(api_key_env)
                
                if not api_key:
                    self.logger.warning(f"API key not found for {provider_name} (env: {api_key_env})")
                    continue
                
                # Create provider instance
                provider = create_provider(
                    provider_name=provider_name,
                    api_key=api_key,
                    model=provider_config.get('model'),
                    max_tokens=provider_config.get('max_tokens', 8000),
                    temperature=provider_config.get('temperature', 0.1),
                    max_retries=provider_config.get('max_retries', 3),
                    timeout=provider_config.get('timeout', 60),
                    verbose=self.verbose
                )
                
                self.providers[provider_name] = provider
                self.health[provider_name] = ProviderHealth(provider_name=provider_name)
                
                if self.verbose:
                    self.logger.info(f"  ✓ Initialized {provider_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {provider_name}: {str(e)}")
    
    def generate(self, prompt: str, task: str = "generation", **kwargs) -> str:
        """
        Generate text response using best available provider
        
        Args:
            prompt: Input prompt
            task: Task description for logging
            **kwargs: Additional parameters
            
        Returns:
            Generated text
            
        Raises:
            LLMProviderError: If all providers fail
        """
        if self.verbose:
            self.logger.info(f"\n🤖 Starting {task} task...")
        
        # Try providers in priority order
        last_error = None
        
        for provider_name in self.provider_priority:
            provider = self.providers.get(provider_name)
            health = self.health.get(provider_name)
            
            if not provider or not health:
                continue
            
            # Check if provider should be tried
            if not health.should_try():
                if self.verbose:
                    self.logger.warning(f"  ⏭️  Skipping {provider_name} (circuit breaker active)")
                continue
            
            # Try provider
            try:
                if self.verbose:
                    self.logger.info(f"  🔄 Trying {provider_name}...")
                
                response = provider.generate(prompt, **kwargs)
                
                # Success!
                health.record_success()
                
                # Track if using backup
                is_primary = provider_name == self.provider_priority[0]
                if is_primary:
                    self.last_primary_success = datetime.now()
                    self.using_backup = False
                else:
                    self.using_backup = True
                
                if self.verbose:
                    self.logger.info(f"  ✅ Success with {provider_name}")
                
                return response
                
            except (RateLimitError, TimeoutError, LLMProviderError) as e:
                last_error = e
                health.record_failure(
                    failure_threshold=self.failure_threshold,
                    circuit_breaker_minutes=self.circuit_breaker_minutes
                )
                
                if self.verbose:
                    self.logger.warning(f"  ❌ {provider_name} failed: {str(e)}")
                
                # Continue to next provider
                continue
            
            except InvalidAPIKeyError as e:
                # Don't retry on invalid API key
                health.record_failure(
                    failure_threshold=1,  # Immediate circuit breaker
                    circuit_breaker_minutes=60  # Longer timeout
                )
                
                self.logger.error(f"  🔑 Invalid API key for {provider_name}")
                last_error = e
                continue
        
        # All providers failed
        error_msg = f"All providers failed. Last error: {str(last_error)}"
        self.logger.error(f"  💥 {error_msg}")
        raise LLMProviderError(error_msg)
    
    def generate_json(self, prompt: str, task: str = "json_generation", **kwargs) -> dict:
        """
        Generate JSON response using best available provider
        
        Args:
            prompt: Input prompt
            task: Task description for logging
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            LLMProviderError: If all providers fail
        """
        if self.verbose:
            self.logger.info(f"\n🤖 Starting {task} task...")
        
        # Try providers in priority order
        last_error = None
        
        for provider_name in self.provider_priority:
            provider = self.providers.get(provider_name)
            health = self.health.get(provider_name)
            
            if not provider or not health:
                continue
            
            # Check if provider should be tried
            if not health.should_try():
                if self.verbose:
                    self.logger.warning(f"  ⏭️  Skipping {provider_name} (circuit breaker active)")
                continue
            
            # Try provider
            try:
                if self.verbose:
                    self.logger.info(f"  🔄 Trying {provider_name}...")
                
                response = provider.generate_json(prompt, **kwargs)
                
                # Success!
                health.record_success()
                
                # Track if using backup
                is_primary = provider_name == self.provider_priority[0]
                if is_primary:
                    self.last_primary_success = datetime.now()
                    self.using_backup = False
                else:
                    self.using_backup = True
                
                if self.verbose:
                    self.logger.info(f"  ✅ Success with {provider_name}")
                
                return response
                
            except (RateLimitError, TimeoutError, LLMProviderError) as e:
                last_error = e
                health.record_failure(
                    failure_threshold=self.failure_threshold,
                    circuit_breaker_minutes=self.circuit_breaker_minutes
                )
                
                if self.verbose:
                    self.logger.warning(f"  ❌ {provider_name} failed: {str(e)}")
                
                # Continue to next provider
                continue
            
            except InvalidAPIKeyError as e:
                # Don't retry on invalid API key
                health.record_failure(
                    failure_threshold=1,  # Immediate circuit breaker
                    circuit_breaker_minutes=60  # Longer timeout
                )
                
                self.logger.error(f"  🔑 Invalid API key for {provider_name}")
                last_error = e
                continue
        
        # All providers failed
        error_msg = f"All providers failed. Last error: {str(last_error)}"
        self.logger.error(f"  💥 {error_msg}")
        raise LLMProviderError(error_msg)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all providers"""
        return {
            'providers': {
                name: health.to_dict()
                for name, health in self.health.items()
            },
            'using_backup': self.using_backup,
            'last_primary_success': self.last_primary_success.isoformat() if self.last_primary_success else None
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all providers"""
        return {
            name: provider.get_usage_stats()
            for name, provider in self.providers.items()
        }
    
    def print_summary(self):
        """Print comprehensive summary of all providers"""
        print("\n" + "="*70)
        print("LLM Manager Summary")
        print("="*70)
        
        # Health status
        print("\n📊 Provider Health:")
        for name, health in self.health.items():
            status = "✅ Healthy" if health.is_healthy else "❌ Unhealthy"
            print(f"  {name}: {status} ({health.success_rate:.1f}% success rate)")
            if health.circuit_breaker_until:
                print(f"    Circuit breaker until: {health.circuit_breaker_until.strftime('%H:%M:%S')}")
        
        # Usage stats
        print("\n💰 Usage Statistics:")
        print_usage_summary(list(self.providers.values()))
        
        # Current status
        print("🎯 Current Status:")
        if self.using_backup:
            print(f"  Using backup provider (primary unavailable)")
        else:
            print(f"  Using primary provider")
        
        print("="*70 + "\n")



def test_llm_manager():
    """Test the LLM manager with multiple providers"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Configuration
    config = {
        'primary_provider': 'gemini',
        'backup_provider': 'groq',
        'failure_threshold': 3,
        'circuit_breaker_minutes': 5,
        'auto_recovery_minutes': 30,
        'gemini': {
            'model': 'gemini-1.5-flash',
            'api_key_env': 'GOOGLE_API_KEY',
            'max_tokens': 8000,
            'temperature': 0.1,
            'max_retries': 3,
            'timeout': 60
        },
        'groq': {
            'model': 'llama-3.1-70b-versatile',
            'api_key_env': 'GROQ_API_KEY',
            'max_tokens': 8000,
            'temperature': 0.1,
            'max_retries': 3,
            'timeout': 60
        }
    }
    
    print("="*70)
    print("Testing LLM Manager")
    print("="*70)
    
    # Initialize manager
    manager = LLMManager(config=config, verbose=True)
    
    # Test simple generation
    try:
        prompt = "What is 2+2? Respond in one sentence."
        response = manager.generate(prompt, task="math_test")
        print(f"\n✅ Response: {response}")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
    
    # Test JSON generation
    try:
        json_prompt = """
        Extract the following information as JSON:
        - name: John Doe
        - age: 30
        - city: New York
        
        Return only the JSON object.
        """
        response = manager.generate_json(json_prompt, task="json_test")
        print(f"\n✅ JSON Response: {response}")
    except Exception as e:
        print(f"\n❌ JSON test failed: {str(e)}")
    
    # Print summary
    manager.print_summary()


if __name__ == "__main__":
    test_llm_manager()
