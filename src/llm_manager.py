"""
LLM Manager Module
Handles multi-provider orchestration, failover logic, and usage tracking
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from src.llm_providers import (
    create_provider, 
    BaseLLMProvider, 
    LLMProviderError,
    RateLimitError,
    TimeoutError
)

class LLMManager:
    """
    Orchestrates multiple LLM providers with automatic failover and recovery
    """
    
    def __init__(self, config: Dict, verbose: bool = True):
        """
        Initialize LLM manager with configuration
        
        Args:
            config: LLM configuration dictionary from config.yaml
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose
        self.logger = logging.getLogger("LLMManager")
        
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
            
        # Initialize providers
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.primary_name = config.get('primary_provider', 'gemini')
        self.backup_name = config.get('backup_provider', 'groq')
        
        # Health tracking
        self.provider_health = {
            self.primary_name: {'active': True, 'failures': 0, 'last_failure': None},
            self.backup_name: {'active': True, 'failures': 0, 'last_failure': None}
        }
        
        # Failover settings
        self.failure_threshold = config.get('failure_threshold', 3)
        self.circuit_breaker_minutes = config.get('circuit_breaker_minutes', 5)
        self.auto_recovery_minutes = config.get('auto_recovery_minutes', 30)
        
        self._initialize_providers()
        
    def _initialize_providers(self):
        """Initialize all configured providers"""
        for name in [self.primary_name, self.backup_name]:
            if name not in self.config:
                continue
                
            prov_config = self.config[name]
            api_key = os.getenv(prov_config.get('api_key_env'))
            
            if not api_key:
                if self.verbose:
                    self.logger.warning(f"API key for {name} not found in environment (tried {prov_config.get('api_key_env')})")
                continue
                
            try:
                self.providers[name] = create_provider(
                    provider_name=name,
                    api_key=api_key,
                    model=prov_config.get('model'),
                    max_tokens=prov_config.get('max_tokens', 8000),
                    temperature=prov_config.get('temperature', 0.1),
                    max_retries=prov_config.get('max_retries', 3),
                    timeout=prov_config.get('timeout', 60),
                    verbose=self.verbose
                )
                if self.verbose:
                    self.logger.info(f"✓ Provider {name} initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize provider {name}: {e}")

    def get_active_provider(self) -> tuple[str, BaseLLMProvider]:
        """
        Get the currently active provider based on health and priority
        
        Returns:
            Tuple of (provider_name, provider_instance)
        """
        # 1. Check if primary should be recovered
        primary_health = self.provider_health[self.primary_name]
        if not primary_health['active'] and primary_health['last_failure']:
            fail_time = primary_health['last_failure']
            if datetime.now() - fail_time > timedelta(minutes=self.circuit_breaker_minutes):
                if self.verbose:
                    self.logger.info(f"🔄 Attempting recovery of primary provider: {self.primary_name}")
                primary_health['active'] = True
                primary_health['failures'] = 0
        
        # 2. Try primary if active
        if primary_health['active'] and self.primary_name in self.providers:
            return self.primary_name, self.providers[self.primary_name]
            
        # 3. Fallback to backup
        if self.backup_name in self.providers:
            if self.verbose and primary_health['active']:
                 self.logger.info(f"⚠️ Primary {self.primary_name} not initialized, using backup {self.backup_name}")
            return self.backup_name, self.providers[self.backup_name]
            
        raise LLMProviderError("No active LLM providers available. Check API keys and configuration.")

    def generate(self, prompt: str, task: str = "general", **kwargs) -> str:
        """
        Generate text response with automatic failover
        """
        try:
            name, provider = self.get_active_provider()
            if self.verbose:
                self.logger.debug(f"[*] Task '{task}': Using provider {name}")
                
            return provider.generate(prompt, **kwargs)
            
        except (RateLimitError, TimeoutError, LLMProviderError) as e:
            self._handle_provider_failure(name)
            
            # If we were using primary, try backup immediately
            if name == self.primary_name and self.backup_name in self.providers:
                if self.verbose:
                    self.logger.warning(f"⚠️ Primary {name} failed: {e}. Switching to backup {self.backup_name}...")
                return self.providers[self.backup_name].generate(prompt, **kwargs)
            
            raise e

    def generate_json(self, prompt: str, task: str = "general", **kwargs) -> Dict:
        """
        Generate JSON response with automatic failover
        """
        try:
            name, provider = self.get_active_provider()
            if self.verbose:
                self.logger.debug(f"[*] Task '{task}': Using provider {name} (JSON)")
                
            return provider.generate_json(prompt, **kwargs)
            
        except (RateLimitError, TimeoutError, LLMProviderError) as e:
            self._handle_provider_failure(name)
            
            # If we were using primary, try backup immediately
            if name == self.primary_name and self.backup_name in self.providers:
                if self.verbose:
                    self.logger.warning(f"⚠️ Primary {name} failed: {e}. Switching to backup {self.backup_name} (JSON)...")
                return self.providers[self.backup_name].generate_json(prompt, **kwargs)
            
            raise e

    def _handle_provider_failure(self, name: str):
        """Track and handle provider failures"""
        health = self.provider_health[name]
        health['failures'] += 1
        health['last_failure'] = datetime.now()
        
        if health['failures'] >= self.failure_threshold:
            health['active'] = False
            self.logger.error(f"🚨 Provider {name} deactivated after {health['failures']} consecutive failures")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Aggregate usage stats from all providers"""
        stats = {}
        for name, provider in self.providers.items():
            stats[name] = provider.get_usage_stats()
        return stats

    def get_health_status() -> Dict[str, Any]:
        """Get current health status of all providers"""
        return {
            'providers': self.provider_health,
            'using_backup': not self.provider_health[self.primary_name]['active'],
            'available_instances': list(self.providers.keys())
        }