#!/usr/bin/env python3
"""
System Audio Setup Helper for KarmaViz

This script helps set up system audio capture for KarmaViz on Linux systems.
It provides instructions and utilities for configuring PulseAudio loopback.
"""

import subprocess
import sys
import os
from modules.audio_handler import list_audio_devices


def check_pulseaudio():
    """Check if PulseAudio is running"""
    try:
        result = subprocess.run(['pulseaudio', '--check'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def list_pulse_sources():
    """List PulseAudio sources"""
    try:
        result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        return []
    except FileNotFoundError:
        return []


def setup_loopback():
    """Set up PulseAudio loopback for system audio capture"""
    print("Setting up PulseAudio loopback for system audio capture...")
    
    try:
        # Load the loopback module
        cmd = ['pactl', 'load-module', 'module-loopback', 'latency_msec=1']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Loopback module loaded successfully!")
            print("You should now see a 'Monitor' device in your audio settings.")
            return True
        else:
            print(f"✗ Failed to load loopback module: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ pactl command not found. Please install PulseAudio utilities.")
        return False


def main():
    print("KarmaViz System Audio Setup Helper")
    print("=" * 40)
    
    # Check PulseAudio
    if not check_pulseaudio():
        print("✗ PulseAudio is not running or not installed.")
        print("Please install and start PulseAudio first.")
        return 1
    
    print("✓ PulseAudio is running")
    
    # List current audio devices
    print("\nCurrent audio input devices:")
    devices = list_audio_devices()
    if devices:
        for device in devices:
            print(f"  {device['id']}: {device['name']} ({device['channels']} channels)")
    else:
        print("  No input devices found")
    
    # List PulseAudio sources
    print("\nPulseAudio sources:")
    sources = list_pulse_sources()
    if sources:
        for source in sources:
            if source.strip():
                print(f"  {source}")
    else:
        print("  Could not list PulseAudio sources")
    
    # Check for monitor sources
    monitor_sources = [s for s in sources if 'monitor' in s.lower()]
    if monitor_sources:
        print(f"\n✓ Found {len(monitor_sources)} monitor source(s) - system audio capture should work!")
    else:
        print("\n! No monitor sources found. Setting up loopback...")
        if setup_loopback():
            print("\nLoopback set up successfully!")
            print("You may need to restart KarmaViz to see the new audio device.")
        else:
            print("\nFailed to set up loopback. Manual setup may be required.")
    
    print("\nInstructions:")
    print("1. Run KarmaViz and check the console output for available audio devices")
    print("2. Look for devices with 'monitor' in the name for system audio capture")
    print("3. If no monitor devices are available, you may need to:")
    print("   - Enable 'Show monitor sources' in your audio settings")
    print("   - Use pavucontrol to configure audio routing")
    print("   - Set up a virtual audio cable")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())#!/usr/bin/env python3
"""
System Audio Setup Helper for KarmaViz

This script helps set up system audio capture for KarmaViz on Linux systems.
It provides instructions and utilities for configuring PulseAudio loopback.
"""

import subprocess
import sys
import os
from modules.audio_handler import list_audio_devices


def check_pulseaudio():
    """Check if PulseAudio is running"""
    try:
        result = subprocess.run(['pulseaudio', '--check'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def list_pulse_sources():
    """List PulseAudio sources"""
    try:
        result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        return []
    except FileNotFoundError:
        return []


def setup_loopback():
    """Set up PulseAudio loopback for system audio capture"""
    print("Setting up PulseAudio loopback for system audio capture...")
    
    try:
        # Load the loopback module
        cmd = ['pactl', 'load-module', 'module-loopback', 'latency_msec=1']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Loopback module loaded successfully!")
            print("You should now see a 'Monitor' device in your audio settings.")
            return True
        else:
            print(f"✗ Failed to load loopback module: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ pactl command not found. Please install PulseAudio utilities.")
        return False


def main():
    print("KarmaViz System Audio Setup Helper")
    print("=" * 40)
    
    # Check PulseAudio
    if not check_pulseaudio():
        print("✗ PulseAudio is not running or not installed.")
        print("Please install and start PulseAudio first.")
        return 1
    
    print("✓ PulseAudio is running")
    
    # List current audio devices
    print("\nCurrent audio input devices:")
    devices = list_audio_devices()
    if devices:
        for device in devices:
            print(f"  {device['id']}: {device['name']} ({device['channels']} channels)")
    else:
        print("  No input devices found")
    
    # List PulseAudio sources
    print("\nPulseAudio sources:")
    sources = list_pulse_sources()
    if sources:
        for source in sources:
            if source.strip():
                print(f"  {source}")
    else:
        print("  Could not list PulseAudio sources")
    
    # Check for monitor sources
    monitor_sources = [s for s in sources if 'monitor' in s.lower()]
    if monitor_sources:
        print(f"\n✓ Found {len(monitor_sources)} monitor source(s) - system audio capture should work!")
    else:
        print("\n! No monitor sources found. Setting up loopback...")
        if setup_loopback():
            print("\nLoopback set up successfully!")
            print("You may need to restart KarmaViz to see the new audio device.")
        else:
            print("\nFailed to set up loopback. Manual setup may be required.")
    
    print("\nInstructions:")
    print("1. Run KarmaViz and check the console output for available audio devices")
    print("2. Look for devices with 'monitor' in the name for system audio capture")
    print("3. If no monitor devices are available, you may need to:")
    print("   - Enable 'Show monitor sources' in your audio settings")
    print("   - Use pavucontrol to configure audio routing")
    print("   - Set up a virtual audio cable")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())