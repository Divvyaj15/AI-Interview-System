"""
Test Script for Sentiment Audio Analysis
Tests emotional state detection, confidence analysis, and stress assessment
"""

import os
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

def test_sentiment_analyzer():
    """Test the sentiment audio analyzer functionality"""
    print("🧪 Testing Sentiment Audio Analyzer")
    print("=" * 50)
    
    try:
        # Import the analyzer
        from sentiment_audio_analyzer import SentimentAudioAnalyzer
        print("✅ Successfully imported SentimentAudioAnalyzer")
        
        # Initialize analyzer
        analyzer = SentimentAudioAnalyzer()
        print("✅ Successfully initialized sentiment analyzer")
        
        # Test feature extraction with synthetic data
        print("\n🔍 Testing sentiment feature extraction...")
        
        # Create a simple test audio file if none exists
        test_audio_path = "test_sentiment_audio.wav"
        if not os.path.exists(test_audio_path):
            print("📁 Creating test audio file for sentiment analysis...")
            create_test_sentiment_audio(test_audio_path)
        
        if os.path.exists(test_audio_path):
            # Test sentiment analysis
            print(f"🎵 Analyzing sentiment from: {test_audio_path}")
            results = analyzer.analyze_sentiment(test_audio_path)
            
            if 'error' not in results:
                print("✅ Sentiment analysis completed successfully!")
                
                # Display primary results
                print(f"\n🎭 Primary Emotion: {results['emotional_states']['primary']}")
                print(f"💪 Confidence Level: {results['confidence_analysis']['level']}")
                print(f"😰 Stress Level: {results['stress_analysis']['level']}")
                print(f"🧠 Emotional Intelligence: {results['emotional_intelligence']:.2f}/10")
                print(f"😊 Overall Sentiment: {results['overall_sentiment']}")
                
                # Show all detected emotions
                print("\n🎭 All Detected Emotions:")
                all_emotions = results['emotional_states']['all_emotions']
                if all_emotions:
                    for emotion, confidence in all_emotions.items():
                        print(f"  • {emotion}: {confidence:.1f}")
                else:
                    print("  • Neutral emotional state")
                
                # Show confidence indicators
                print("\n💪 Confidence Indicators:")
                confidence_indicators = results['confidence_analysis']['indicators']
                for indicator in confidence_indicators:
                    print(f"  • {indicator}")
                
                # Show stress markers
                print("\n😰 Stress Markers:")
                stress_markers = results['stress_analysis']['markers']
                for marker in stress_markers:
                    print(f"  • {marker}")
                
                # Show recommendations
                print("\n💡 Recommendations:")
                recommendations = results['recommendations']
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
                
                # Show feature breakdown
                print("\n🔍 Sentiment Features Breakdown:")
                features = results['features']
                for feature, score in features.items():
                    feature_name = feature.replace("_", " ").title()
                    print(f"  • {feature_name}: {score:.2f}/10")
                
            else:
                print(f"❌ Sentiment analysis failed: {results['error']}")
        else:
            print("❌ Test audio file not found")
        
        # Test model training
        print("\n🤖 Testing sentiment model training...")
        try:
            # Generate small training dataset
            training_data = generate_sentiment_training_data(n_samples=100)
            print(f"✅ Generated {len(training_data)} training samples")
            
            # Train model
            analyzer.train_sentiment_model(training_data['features'], training_data['labels'])
            print("✅ Sentiment model training completed")
            
            # Test prediction
            test_features = training_data['features'][0]
            prediction = analyzer.predict_emotion(test_features)
            print(f"✅ Emotion prediction test: {prediction}")
            
        except Exception as e:
            print(f"⚠️ Model training test failed: {e}")
            print("This is expected if dependencies are missing")
        
        print("\n🎉 Sentiment Audio Analyzer test completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install librosa scikit-learn numpy pandas scipy joblib soundfile")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def create_test_sentiment_audio(file_path: str):
    """Create test audio file with emotional variations for sentiment testing"""
    try:
        import numpy as np
        import soundfile as sf
        
        # Generate audio with emotional variations
        sample_rate = 22050
        duration = 5  # 5 seconds
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create base audio with emotional variations
        # Start calm, then become more expressive, then stressed
        base_audio = 0.3 * np.sin(2 * np.pi * 200 * t)
        
        # Add emotional variations
        # First 2 seconds: calm and stable
        base_audio[:int(2 * sample_rate)] *= 0.8
        
        # Middle 2 seconds: more expressive and enthusiastic
        middle_start = int(2 * sample_rate)
        middle_end = int(4 * sample_rate)
        base_audio[middle_start:middle_end] *= 1.5
        base_audio[middle_start:middle_end] += 0.2 * np.sin(2 * np.pi * 300 * t[middle_start:middle_end])
        
        # Last second: some stress indicators (variations)
        stress_start = int(4 * sample_rate)
        base_audio[stress_start:] *= 1.2
        base_audio[stress_start:] += 0.3 * np.sin(2 * np.pi * 150 * t[stress_start:])
        
        # Add some noise to simulate real-world conditions
        base_audio += 0.05 * np.random.normal(0, 1, len(base_audio))
        
        # Save audio file
        sf.write(file_path, base_audio, sample_rate)
        print(f"✅ Test sentiment audio file created: {file_path}")
        
    except ImportError:
        print("⚠️ soundfile not available, creating empty test file")
        # Create empty file as fallback
        with open(file_path, 'w') as f:
            f.write("test")
    except Exception as e:
        print(f"⚠️ Could not create test audio: {e}")

def generate_sentiment_training_data(n_samples: int = 100):
    """Generate synthetic training data for sentiment analysis"""
    print(f"Generating {n_samples} synthetic sentiment training samples...")
    
    features_list = []
    labels_list = []
    
    # Define emotional states and their characteristics
    emotion_characteristics = {
        'Confident': {
            'pitch_stability': (8.0, 10.0),
            'speech_rate_consistency': (8.0, 10.0),
            'formant_stability': (8.0, 10.0),
            'jitter': (8.0, 10.0),
            'shimmer': (8.0, 10.0)
        },
        'Anxious': {
            'pitch_stability': (2.0, 5.0),
            'speech_rate_consistency': (2.0, 5.0),
            'voice_tremor': (7.0, 10.0),
            'jitter': (2.0, 5.0),
            'shimmer': (2.0, 5.0)
        },
        'Enthusiastic': {
            'energy_variation': (8.0, 10.0),
            'volume_fluctuation': (8.0, 10.0),
            'emotional_intensity': (8.0, 10.0),
            'spectral_centroid_variance': (7.0, 10.0)
        },
        'Calm': {
            'pitch_stability': (8.0, 10.0),
            'speech_rate_consistency': (8.0, 10.0),
            'stress_markers': (0.0, 3.0),
            'voice_tremor': (0.0, 3.0)
        }
    }
    
    # Generate samples for each emotion
    samples_per_emotion = n_samples // len(emotion_characteristics)
    
    for emotion, characteristics in emotion_characteristics.items():
        for _ in range(samples_per_emotion):
            features = {}
            
            # Generate core features based on emotion
            for feature, (min_val, max_val) in characteristics.items():
                features[feature] = np.random.uniform(min_val, max_val)
            
            # Generate other features with realistic correlations
            features['energy_variation'] = np.random.uniform(3, 8)
            features['pitch_stability'] = features.get('pitch_stability', 5)
            features['speech_rate_consistency'] = features.get('speech_rate_consistency', 5)
            features['volume_fluctuation'] = features.get('volume_fluctuation', 5)
            features['pause_patterns'] = np.random.uniform(4, 8)
            features['voice_tremor'] = features.get('voice_tremor', 5)
            features['spectral_centroid_variance'] = features.get('spectral_centroid_variance', 5)
            features['mfcc_variation'] = np.random.uniform(3, 8)
            features['harmonic_ratio'] = np.random.uniform(4, 9)
            features['noise_level'] = np.random.uniform(6, 9)
            features['formant_stability'] = features.get('formant_stability', 5)
            features['jitter'] = features.get('jitter', 5)
            features['shimmer'] = features.get('shimmer', 5)
            
            # Calculate derived features
            features['emotional_intensity'] = np.mean([
                features.get('energy_variation', 5),
                features.get('volume_fluctuation', 5),
                features.get('spectral_centroid_variance', 5),
                features.get('mfcc_variation', 5)
            ])
            
            features['confidence_indicators'] = np.mean([
                features.get('pitch_stability', 5),
                features.get('speech_rate_consistency', 5),
                features.get('formant_stability', 5),
                features.get('jitter', 5),
                features.get('shimmer', 5)
            ])
            
            features['stress_markers'] = np.mean([
                10 - features.get('pitch_stability', 5),
                10 - features.get('speech_rate_consistency', 5),
                features.get('voice_tremor', 5),
                features.get('jitter', 5),
                features.get('shimmer', 5)
            ])
            
            # Ensure all values are within 0-10 range
            for key in features:
                features[key] = np.clip(features[key], 0, 10)
            
            features_list.append(features)
            labels_list.append(emotion)
    
    print(f"Generated {len(features_list)} sentiment training samples")
    return {
        'features': features_list,
        'labels': labels_list
    }

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🔍 Testing Sentiment Analysis Dependencies")
    print("=" * 40)
    
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
    print("🚀 Sentiment Audio Analysis - Test Suite")
    print("=" * 60)
    
    # Test dependencies first
    deps_ok = test_dependencies()
    
    if deps_ok:
        print("\n" + "=" * 60)
        # Test the sentiment analyzer
        test_sentiment_analyzer()
    else:
        print("\n❌ Cannot run tests due to missing dependencies")
        print("Please install missing packages and try again")

if __name__ == "__main__":
    main()
