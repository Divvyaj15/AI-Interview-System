#!/usr/bin/env python3
"""
Audio test script for AI Interview System
Run this to test if text-to-speech is working
"""

import pygame
import asyncio
import edge_tts
import tempfile
import os

def test_pygame_audio():
    """Test if pygame can initialize audio"""
    print("🔊 Testing Pygame Audio...")
    
    try:
        # Try to initialize pygame mixer
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        print("✅ Pygame mixer initialized successfully")
        
        # Check if audio device is available
        if pygame.mixer.get_init():
            print("✅ Audio device is available")
            pygame.mixer.quit()
            return True
        else:
            print("❌ Audio device not available")
            return False
            
    except pygame.error as e:
        print(f"❌ Pygame audio error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

async def test_edge_tts():
    """Test Edge TTS functionality"""
    print("\n🎤 Testing Edge TTS...")
    
    try:
        # Test with a simple message
        text = "Hello, this is a test of the text to speech system."
        voice = "en-US-GuyNeural"
        
        communicate = edge_tts.Communicate(text, voice)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            await communicate.save(tmp_file.name)
            
            print(f"✅ TTS file created: {tmp_file.name}")
            print(f"📁 File size: {os.path.getsize(tmp_file.name)} bytes")
            
            # Try to play it
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                pygame.mixer.music.load(tmp_file.name)
                pygame.mixer.music.play()
                
                print("🔊 Playing audio... (you should hear 'Hello, this is a test...')")
                
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
                pygame.mixer.quit()
                print("✅ Audio playback completed")
                
            except pygame.error as e:
                print(f"❌ Audio playback failed: {e}")
                return False
            
            # Clean up
            os.unlink(tmp_file.name)
            
        return True
        
    except Exception as e:
        print(f"❌ Edge TTS error: {e}")
        return False

def test_system_audio():
    """Test basic system audio"""
    print("\n🔊 Testing System Audio...")
    
    try:
        # Try to play a simple beep
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        
        # Generate a simple beep sound
        sample_rate = 22050
        duration = 0.5  # seconds
        frequency = 440  # Hz (A note)
        
        # Generate sine wave
        import numpy as np
        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit audio
        audio_data = (wave * 32767).astype(np.int16)
        
        # Save as temporary WAV file
        import wave as wave_module
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            with wave_module.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # Try to play it
            pygame.mixer.music.load(tmp_file.name)
            pygame.mixer.music.play()
            
            print("🔊 Playing test beep... (you should hear a beep sound)")
            
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            pygame.mixer.quit()
            os.unlink(tmp_file.name)
            
            print("✅ System audio test completed")
            return True
            
    except Exception as e:
        print(f"❌ System audio test failed: {e}")
        return False

def main():
    """Run all audio tests"""
    print("🎵 AI Interview System - Audio Test")
    print("=" * 50)
    
    # Test 1: Pygame audio
    pygame_ok = test_pygame_audio()
    
    # Test 2: System audio
    system_ok = test_system_audio()
    
    # Test 3: Edge TTS
    tts_ok = asyncio.run(test_edge_tts())
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Pygame Audio: {'✅ PASS' if pygame_ok else '❌ FAIL'}")
    print(f"System Audio: {'✅ PASS' if system_ok else '❌ FAIL'}")
    print(f"Edge TTS: {'✅ PASS' if tts_ok else '❌ FAIL'}")
    
    if pygame_ok and system_ok and tts_ok:
        print("\n🎉 All audio tests passed! TTS should work in the app.")
    else:
        print("\n⚠️  Some audio tests failed. Here are solutions:")
        
        if not pygame_ok:
            print("🔧 Pygame issues:")
            print("   - Check if audio device is connected")
            print("   - Try updating audio drivers")
            print("   - Restart your computer")
            
        if not system_ok:
            print("🔧 System audio issues:")
            print("   - Check system volume")
            print("   - Unmute your speakers/headphones")
            print("   - Try different audio output device")
            
        if not tts_ok:
            print("🔧 TTS issues:")
            print("   - Check internet connection")
            print("   - Try restarting the app")
            print("   - The app will work without audio (text-only mode)")

if __name__ == "__main__":
    main()
