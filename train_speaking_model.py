"""
Training Script for Speaking Skills Model
Generates synthetic training data and trains the ML model
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from speaking_skills_analyzer import SpeakingSkillsAnalyzer

def generate_synthetic_training_data(n_samples: int = 1000) -> tuple:
    """
    Generate synthetic training data for speaking skills assessment
    
    Args:
        n_samples: Number of training samples to generate
        
    Returns:
        Tuple of (features_list, labels_list)
    """
    print(f"Generating {n_samples} synthetic training samples...")
    
    features_list = []
    labels_list = []
    
    # Define speaking skill levels and their characteristics
    skill_levels = {
        'Exceptional': {
            'speech_rate': (7.0, 8.5),
            'clarity_score': (8.5, 10.0),
            'confidence_score': (8.5, 10.0),
            'fluency_score': (8.5, 10.0),
            'articulation_score': (8.5, 10.0)
        },
        'Excellent': {
            'speech_rate': (6.0, 8.0),
            'clarity_score': (7.0, 9.0),
            'confidence_score': (7.0, 9.0),
            'fluency_score': (7.0, 9.0),
            'articulation_score': (7.0, 9.0)
        },
        'Good': {
            'speech_rate': (5.0, 7.5),
            'clarity_score': (5.5, 8.0),
            'confidence_score': (5.5, 8.0),
            'fluency_score': (5.5, 8.0),
            'articulation_score': (5.5, 8.0)
        },
        'Fair': {
            'speech_rate': (4.0, 6.5),
            'clarity_score': (4.0, 6.5),
            'confidence_score': (4.0, 6.5),
            'fluency_score': (4.0, 6.5),
            'articulation_score': (4.0, 6.5)
        },
        'Needs Improvement': {
            'speech_rate': (2.0, 5.5),
            'clarity_score': (2.0, 5.0),
            'confidence_score': (2.0, 5.0),
            'fluency_score': (2.0, 5.0),
            'articulation_score': (2.0, 5.0)
        }
    }
    
    # Generate samples for each skill level
    samples_per_level = n_samples // len(skill_levels)
    
    for skill_level, characteristics in skill_levels.items():
        for _ in range(samples_per_level):
            features = {}
            
            # Generate core features based on skill level
            for feature, (min_val, max_val) in characteristics.items():
                features[feature] = np.random.uniform(min_val, max_val)
            
            # Generate other features with realistic correlations
            features['pause_frequency'] = 10 - features.get('fluency_score', 5) + np.random.normal(0, 1)
            features['energy_level'] = features.get('confidence_score', 5) + np.random.normal(0, 1.5)
            features['pitch_variation'] = features.get('articulation_score', 5) + np.random.normal(0, 1)
            features['volume_consistency'] = features.get('confidence_score', 5) + np.random.normal(0, 1)
            features['pace_consistency'] = features.get('fluency_score', 5) + np.random.normal(0, 1)
            features['emphasis_effectiveness'] = features.get('articulation_score', 5) + np.random.normal(0, 1)
            
            # Ensure all values are within 0-10 range
            for key in features:
                features[key] = np.clip(features[key], 0, 10)
            
            # Calculate overall score
            weights = {
                'speech_rate': 0.1,
                'pause_frequency': 0.1,
                'clarity_score': 0.15,
                'confidence_score': 0.15,
                'energy_level': 0.1,
                'pitch_variation': 0.1,
                'volume_consistency': 0.1,
                'fluency_score': 0.1,
                'articulation_score': 0.1
            }
            
            overall_score = 0
            for feature, weight in weights.items():
                if feature in features:
                    overall_score += features[feature] * weight
            
            features['overall_communication_score'] = min(overall_score, 10)
            
            features_list.append(features)
            labels_list.append(skill_level)
    
    # Add some random samples for variety
    remaining_samples = n_samples - len(features_list)
    for _ in range(remaining_samples):
        features = {}
        for feature in ['speech_rate', 'pause_frequency', 'clarity_score', 'confidence_score',
                       'energy_level', 'pitch_variation', 'volume_consistency', 'fluency_score',
                       'articulation_score', 'pace_consistency', 'emphasis_effectiveness']:
            features[feature] = np.random.uniform(2, 9)
        
        # Calculate overall score
        weights = {
            'speech_rate': 0.1,
            'pause_frequency': 0.1,
            'clarity_score': 0.15,
            'confidence_score': 0.15,
            'energy_level': 0.1,
            'pitch_variation': 0.1,
            'volume_consistency': 0.1,
            'fluency_score': 0.1,
            'articulation_score': 0.1
        }
        
        overall_score = 0
        for feature, weight in weights.items():
            if feature in features:
                overall_score += features[feature] * weight
        
        features['overall_communication_score'] = min(overall_score, 10)
        
        # Assign label based on overall score
        if overall_score >= 8.5:
            label = 'Exceptional'
        elif overall_score >= 7.0:
            label = 'Excellent'
        elif overall_score >= 5.5:
            label = 'Good'
        elif overall_score >= 4.0:
            label = 'Fair'
        else:
            label = 'Needs Improvement'
        
        features_list.append(features)
        labels_list.append(label)
    
    print(f"Generated {len(features_list)} training samples")
    return features_list, labels_list

def create_sample_audio_files():
    """Create sample audio files for testing"""
    print("Creating sample audio files for testing...")
    
    # Create audio directory if it doesn't exist
    audio_dir = Path("audio/samples")
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate simple sine wave audio files for testing
    import librosa
    import soundfile as sf
    
    # Sample 1: Good speaking (moderate frequency, consistent)
    sample_rate = 22050
    duration = 5  # 5 seconds
    
    # Good speaking sample
    t = np.linspace(0, duration, int(sample_rate * duration))
    good_audio = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.sin(2 * np.pi * 400 * t)
    good_audio += 0.05 * np.random.normal(0, 1, len(good_audio))
    
    sf.write(audio_dir / "good_speaking.wav", good_audio, sample_rate)
    
    # Poor speaking sample (more variation, less consistent)
    poor_audio = 0.2 * np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 300 * t)
    poor_audio += 0.2 * np.random.normal(0, 1, len(poor_audio))
    
    sf.write(audio_dir / "poor_speaking.wav", poor_audio, sample_rate)
    
    print("Sample audio files created in audio/samples/")
    return audio_dir

def train_model():
    """Train the speaking skills model"""
    print("Starting model training...")
    
    # Initialize analyzer
    analyzer = SpeakingSkillsAnalyzer()
    
    # Generate training data
    features_list, labels_list = generate_synthetic_training_data(n_samples=2000)
    
    # Train the model
    print("Training the model...")
    analyzer.train_model(features_list, labels_list)
    
    print("Model training completed!")
    
    # Test the model with sample data
    print("\nTesting the model...")
    test_features = features_list[0]
    prediction = analyzer.predict_assessment(test_features)
    print(f"Sample prediction: {prediction}")
    
    return analyzer

def test_with_real_audio():
    """Test the model with real audio files if available"""
    print("\nTesting with real audio files...")
    
    analyzer = SpeakingSkillsAnalyzer()
    
    # Check if we have any audio files
    audio_dir = Path("audio")
    if audio_dir.exists():
        audio_files = list(audio_dir.rglob("*.wav")) + list(audio_dir.rglob("*.mp3"))
        
        if audio_files:
            print(f"Found {len(audio_files)} audio files for testing")
            
            for audio_file in audio_files[:3]:  # Test first 3 files
                try:
                    print(f"\nAnalyzing: {audio_file.name}")
                    results = analyzer.analyze_speaking_skills(str(audio_file))
                    
                    print(f"Overall Score: {results['overall_score']:.2f}/10")
                    print(f"Assessment: {results['assessment']}")
                    
                    # Show top 3 strengths and areas for improvement
                    breakdown = results['score_breakdown']
                    sorted_skills = sorted(breakdown.items(), key=lambda x: x[1]['score'], reverse=True)
                    
                    print("Top 3 Strengths:")
                    for skill, details in sorted_skills[:3]:
                        print(f"  {skill}: {details['score']:.2f}/10 ({details['grade']})")
                    
                    print("Areas for Improvement:")
                    for skill, details in sorted_skills[-3:]:
                        print(f"  {skill}: {details['score']:.2f}/10 ({details['grade']})")
                    
                except Exception as e:
                    print(f"Error analyzing {audio_file.name}: {e}")
        else:
            print("No audio files found for testing")
    else:
        print("Audio directory not found")

def main():
    """Main training and testing function"""
    print("🚀 Speaking Skills Model Training and Testing")
    print("=" * 50)
    
    try:
        # Create sample audio files
        audio_dir = create_sample_audio_files()
        
        # Train the model
        analyzer = train_model()
        
        # Test with real audio
        test_with_real_audio()
        
        print("\n✅ Training and testing completed successfully!")
        print(f"Model saved to: {analyzer.model_path}")
        print(f"Sample audio files created in: {audio_dir}")
        
        print("\n📊 Model Features:")
        for feature in analyzer.feature_names:
            print(f"  - {feature}")
        
        print("\n🎯 Usage:")
        print("from utils.speaking_skills_analyzer import SpeakingSkillsAnalyzer")
        print("analyzer = SpeakingSkillsAnalyzer()")
        print("results = analyzer.analyze_speaking_skills('path/to/audio.wav')")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
