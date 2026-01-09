import asyncio
import edge_tts
import pygame
import tempfile
import os
import time


async def speak_edge_tts(text, voice="en-US-AriaNeural", rate="+0%", pitch="+0Hz"):
    """
    High-quality TTS using Microsoft Edge voices

    Popular voices:
    - en-US-AriaNeural (female)
    - en-US-GuyNeural (male)
    - en-GB-SoniaNeural (British female)
    - en-AU-NatashaNeural (Australian female)
    """
    tmp_file_path = None
    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)

        # Create a unique temporary file path
        timestamp = int(time.time() * 1000)
        tmp_file_path = f"temp_audio_{timestamp}.mp3"
        
        # Save the audio
        await communicate.save(tmp_file_path)
        
        # Wait a moment to ensure file is written
        time.sleep(0.1)

        # Play using pygame
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.music.load(tmp_file_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)

            pygame.mixer.quit()
            print(f"✅ TTS played successfully: '{text[:50]}...'")
            
        except pygame.error as e:
            print(f"❌ Pygame audio error: {e}")
            print("💡 This might be due to:")
            print("   - No audio output device connected")
            print("   - Audio device is muted")
            print("   - Audio drivers need updating")
            print("   - System volume is too low")

    except Exception as e:
        print(f"❌ Edge-TTS Error: {e}")
        print(f"Text that failed: {text[:100]}...")
    
    finally:
        # Clean up the temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass  # Ignore cleanup errors


def speak_text(text, voice="en-US-GuyNeural", rate="+0%", pitch="+0Hz"):
    """Synchronous wrapper for Edge-TTS with better error handling"""
    try:
        print(f"🔊 Attempting to speak: '{text[:50]}...'")
        
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an event loop, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, speak_edge_tts(text, voice, rate, pitch))
                future.result()
        else:
            # If no event loop is running, we can use asyncio.run
            asyncio.run(speak_edge_tts(text, voice, rate, pitch))
            
    except RuntimeError:
        # Fallback: create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(speak_edge_tts(text, voice, rate, pitch))
        finally:
            loop.close()
    except Exception as e:
        print(f"❌ Text-to-speech failed: {e}")
        print("💡 Continuing without audio...")
        print("📝 You can still read the questions on screen")
