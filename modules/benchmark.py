"""Benchmarking decorator and utilities for performance monitoring."""

import time
import functools
from collections import defaultdict, deque
from typing import Dict, Callable, Any, Optional
import threading


class PerformanceMonitor:
    """Thread-safe performance monitoring system for tracking function execution times."""
    
    def __init__(self, max_samples: int = 100):
        """Initialize the performance monitor.
        
        Args:
            max_samples: Maximum number of samples to keep for each function
        """
        self.max_samples = max_samples
        self._stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self._lock = threading.Lock()
        self._enabled = True
    
    def record_time(self, func_name: str, execution_time: float) -> None:
        """Record execution time for a function.
        
        Args:
            func_name: Name of the function
            execution_time: Execution time in seconds
        """
        if not self._enabled:
            return
            
        with self._lock:
            self._stats[func_name].append(execution_time)
    
    def get_stats(self, func_name: str) -> Dict[str, float]:
        """Get statistics for a specific function.
        
        Args:
            func_name: Name of the function
            
        Returns:
            Dictionary containing min, max, avg, and latest execution times
        """
        with self._lock:
            times = list(self._stats[func_name])
            
        if not times:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "latest": 0.0, "count": 0}
        
        return {
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / len(times),
            "latest": times[-1],
            "count": len(times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all monitored functions.
        
        Returns:
            Dictionary mapping function names to their statistics
        """
        with self._lock:
            func_names = list(self._stats.keys())
        
        return {name: self.get_stats(name) for name in func_names}
    
    def clear_stats(self, func_name: Optional[str] = None) -> None:
        """Clear statistics for a specific function or all functions.
        
        Args:
            func_name: Name of the function to clear, or None to clear all
        """
        with self._lock:
            if func_name:
                self._stats[func_name].clear()
            else:
                self._stats.clear()
    
    def enable(self) -> None:
        """Enable performance monitoring."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable performance monitoring."""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if performance monitoring is enabled."""
        return self._enabled


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def benchmark(func_name: Optional[str] = None, enabled: bool = True) -> Callable:
    """Decorator to benchmark function execution time.
    
    Args:
        func_name: Custom name for the function (defaults to function.__name__)
        enabled: Whether benchmarking is enabled for this function
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        name = func_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not enabled or not _performance_monitor.is_enabled():
                return func(*args, **kwargs)
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                _performance_monitor.record_time(name, execution_time)
        
        return wrapper
    return decorator


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


def print_performance_stats(func_name: Optional[str] = None) -> None:
    """Print performance statistics to console.
    
    Args:
        func_name: Name of specific function to print, or None for all functions
    """
    if func_name:
        stats = _performance_monitor.get_stats(func_name)
        if stats["count"] > 0:
            print(f"\n=== Performance Stats for {func_name} ===")
            print(f"Count: {stats['count']}")
            print(f"Latest: {stats['latest']*1000:.2f}ms")
            print(f"Average: {stats['avg']*1000:.2f}ms")
            print(f"Min: {stats['min']*1000:.2f}ms")
            print(f"Max: {stats['max']*1000:.2f}ms")
        else:
            print(f"No performance data available for {func_name}")
    else:
        all_stats = _performance_monitor.get_all_stats()
        if all_stats:
            print("\n=== Performance Stats for All Functions ===")
            for name, stats in all_stats.items():
                if stats["count"] > 0:
                    print(f"{name:25} | Latest: {stats['latest']*1000:6.2f}ms | "
                          f"Avg: {stats['avg']*1000:6.2f}ms | "
                          f"Min: {stats['min']*1000:6.2f}ms | "
                          f"Max: {stats['max']*1000:6.2f}ms | "
                          f"Count: {stats['count']:4d}")
        else:
            print("No performance data available")


def get_performance_summary() -> str:
    """Get a formatted string summary of performance statistics.
    
    Returns:
        Formatted performance summary string
    """
    all_stats = _performance_monitor.get_all_stats()
    if not all_stats:
        return "No performance data available"
    
    lines = ["Performance Summary:"]
    for name, stats in all_stats.items():
        if stats["count"] > 0:
            lines.append(f"  {name}: {stats['latest']*1000:.1f}ms (avg: {stats['avg']*1000:.1f}ms)")
    
    return "\n".join(lines)