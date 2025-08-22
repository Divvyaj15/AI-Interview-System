"""
Test Script for Speaking Skills Analyzer
Simple test to verify the analyzer works correctly
"""

import os
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

def test_speaking_skills_analyzer():
    """Test the speaking skills analyzer functionality"""
    print("🧪 Testing Speaking Skills Analyzer")
    print("=" * 40)
    
    try:
        # Import the analyzer
        from speaking_skills_analyzer import SpeakingSkillsAnalyzer
        print("✅ Successfully imported SpeakingSkillsAnalyzer")
        
        # Initialize analyzer
        analyzer = SpeakingSkillsAnalyzer()
        print("✅ Successfully initialized analyzer")
        
        # Test feature extraction with synthetic data
        print("\n🔍 Testing feature extraction...")
        
        # Create a simple test audio file if none exists
        test_audio_path = "test_audio.wav"
        if not os.path.exists(test_audio_path):
            print("📁 Creating test audio file...")
            create_test_audio(test_audio_path)
        
        if os.path.exists(test_audio_path):
            # Test analysis
            print(f"🎵 Analyzing test audio: {test_audio_path}")
            results = analyzer.analyze_speaking_skills(test_audio_path)
            
            if 'error' not in results:
                print("✅ Analysis completed successfully!")
                print(f"📊 Overall Score: {results['overall_score']:.2f}/10")
                print(f"🏆 Assessment: {results['assessment']}")
                
                # Show feature breakdown
                print("\n📈 Feature Breakdown:")
                breakdown = results['score_breakdown']
                for skill, details in breakdown.items():
                    print(f"  {skill}: {details['score']:.2f}/10 ({details['grade']})")
                
                # Show feedback
                print("\n💡 Feedback:")
                for area, advice in results['feedback'].items():
                    area_name = area.replace("_", " ").title()
                    print(f"  {area_name}: {advice}")
                
            else:
                print(f"❌ Analysis failed: {results['error']}")
        else:
            print("❌ Test audio file not found")
        
        # Test model training
        print("\n🤖 Testing model training...")
        try:
            # Generate small training dataset
            from train_speaking_model import generate_synthetic_training_data
            
            features_list, labels_list = generate_synthetic_training_data(n_samples=100)
            print(f"✅ Generated {len(features_list)} training samples")
            
            # Train model
            analyzer.train_model(features_list, labels_list)
            print("✅ Model training completed")
            
            # Test prediction
            test_features = features_list[0]
            prediction = analyzer.predict_assessment(test_features)
            print(f"✅ Prediction test: {prediction}")
            
        except Exception as e:
            print(f"⚠️ Model training test failed: {e}")
            print("This is expected if dependencies are missing")
        
        print("\n🎉 Speaking Skills Analyzer test completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install librosa scikit-learn numpy pandas scipy joblib soundfile")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def create_test_audio(file_path: str):
    """Create a simple test audio file"""
    try:
        import numpy as np
        import soundfile as sf
        
        # Generate simple sine wave audio
        sample_rate = 22050
        duration = 3  # 3 seconds
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create speech-like audio with variations
        audio = 0.3 * np.sin(2 * np.pi * 200 * t)  # Base frequency
        audio += 0.1 * np.sin(2 * np.pi * 400 * t)  # Harmonic
        audio += 0.05 * np.random.normal(0, 1, len(audio))  # Noise
        
        # Add some variation to simulate speech
        for i in range(0, len(audio), sample_rate // 2):  # Every 0.5 seconds
            if i + sample_rate // 4 < len(audio):
                audio[i:i + sample_rate // 4] *= 1.5  # Emphasize some parts
        
        # Save audio file
        sf.write(file_path, audio, sample_rate)
        print(f"✅ Test audio file created: {file_path}")
        
    except ImportError:
        print("⚠️ soundfile not available, creating empty test file")
        # Create empty file as fallback
        with open(file_path, 'w') as f:
            f.write("test")
    except Exception as e:
        print(f"⚠️ Could not create test audio: {e}")

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🔍 Testing Dependencies")
    print("=" * 30)
    
    dependencies = [
        ('numpy', 'Numerical computing'),
        ('pandas', 'Data manipulation'),
        ('librosa', 'Audio processing'),
        ('sklearn', 'Machine learning'),
        ('scipy', 'Scientific computing'),
        ('joblib', 'Model persistence'),
        ('soundfile', 'Audio file handling')
    ]
    
    all_available = True
    
    for package, description in dependencies:
        try:
            if package == 'sklearn':
                import sklearn
                print(f"✅ {package} - {description}")
            else:
                __import__(package)
                print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description} (MISSING)")
            all_available = False
    
    if all_available:
        print("\n🎉 All dependencies are available!")
    else:
        print("\n⚠️ Some dependencies are missing.")
        print("Install missing packages with:")
        print("pip install librosa scikit-learn numpy pandas scipy joblib soundfile")
    
    return all_available

def main():
    """Main test function"""
    print("🚀 Speaking Skills Analyzer - Test Suite")
    print("=" * 50)
    
    # Test dependencies first
    deps_ok = test_dependencies()
    
    if deps_ok:
        print("\n" + "=" * 50)
        # Test the analyzer
        test_speaking_skills_analyzer()
    else:
        print("\n❌ Cannot run tests due to missing dependencies")
        print("Please install missing packages and try again")

if __name__ == "__main__":
    main()
