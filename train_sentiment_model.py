"""
Train Sentiment Audio Analyzer Model
Creates and trains the sentiment_model.pkl using available audio files
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add the parent directory to path to import directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from the file to avoid dependency issues
from utils.sentiment_audio_analyzer import SentimentAudioAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_audio_files(root_dir="audio"):
    """Find all audio files in the directory"""
    audio_files = []
    audio_path = Path(root_dir)
    
    if not audio_path.exists():
        logger.warning(f"Audio directory '{root_dir}' not found")
        return audio_files
    
    # Find all WAV files
    for audio_file in audio_path.rglob("*.wav"):
        if audio_file.exists():
            audio_files.append(str(audio_file))
    
    logger.info(f"Found {len(audio_files)} audio files")
    return audio_files


def generate_training_data_from_audio(audio_files, analyzer):
    """Generate training data from existing audio files"""
    training_features = []
    training_labels = []
    
    logger.info("Extracting features from audio files...")
    
    for i, audio_file in enumerate(audio_files):
        try:
            logger.info(f"Processing ({i+1}/{len(audio_files)}): {Path(audio_file).name}")
            
            # Extract features
            features = analyzer.extract_sentiment_features(audio_file)
            
            # Use rule-based analysis to get emotion label
            emotional_analysis = analyzer._analyze_emotional_states(features)
            primary_emotion = emotional_analysis['primary']
            
            # Create feature vector (only the sentiment features)
            feature_vector = [features.get(feat, 5.0) for feat in analyzer.sentiment_features]
            
            training_features.append(feature_vector)
            training_labels.append(primary_emotion)
            
            logger.info(f"  → Emotion: {primary_emotion}")
            
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
            continue
    
    return training_features, training_labels


def generate_synthetic_training_data(num_samples=500):
    """Generate synthetic training data based on emotional profiles"""
    logger.info(f"Generating {num_samples} synthetic training samples...")
    
    training_features = []
    training_labels = []
    
    emotion_profiles = {
        'Confident': {
            'energy_variation': (6, 2),
            'pitch_stability': (8, 1),
            'speech_rate_consistency': (8, 1),
            'volume_fluctuation': (6, 1.5),
            'pause_patterns': (7, 1),
            'voice_tremor': (2, 1),
            'confidence_indicators': (8, 1),
            'stress_markers': (2, 1)
        },
        'Calm': {
            'energy_variation': (4, 1),
            'pitch_stability': (9, 0.5),
            'speech_rate_consistency': (9, 0.5),
            'volume_fluctuation': (3, 1),
            'pause_patterns': (8, 1),
            'voice_tremor': (1, 0.5),
            'confidence_indicators': (7, 1),
            'stress_markers': (1, 0.5)
        },
        'Enthusiastic': {
            'energy_variation': (9, 1),
            'pitch_stability': (7, 1),
            'speech_rate_consistency': (7, 1),
            'volume_fluctuation': (8, 1),
            'pause_patterns': (6, 1),
            'voice_tremor': (3, 1),
            'confidence_indicators': (7, 1),
            'stress_markers': (3, 1)
        },
        'Stressed': {
            'energy_variation': (7, 1.5),
            'pitch_stability': (4, 1.5),
            'speech_rate_consistency': (4, 1.5),
            'volume_fluctuation': (7, 1.5),
            'pause_patterns': (5, 1),
            'voice_tremor': (8, 1),
            'confidence_indicators': (4, 1),
            'stress_markers': (8, 1)
        },
        'Professional': {
            'energy_variation': (5, 1),
            'pitch_stability': (8, 0.8),
            'speech_rate_consistency': (8, 0.8),
            'volume_fluctuation': (5, 1),
            'pause_patterns': (8, 1),
            'voice_tremor': (2, 0.8),
            'confidence_indicators': (8, 0.8),
            'stress_markers': (2, 0.8)
        },
        'Nervous': {
            'energy_variation': (6, 1.5),
            'pitch_stability': (3, 1.5),
            'speech_rate_consistency': (3, 1.5),
            'volume_fluctuation': (6, 1.5),
            'pause_patterns': (4, 1),
            'voice_tremor': (7, 1),
            'confidence_indicators': (3, 1),
            'stress_markers': (7, 1)
        },
        'Anxious': {
            'energy_variation': (7, 1.5),
            'pitch_stability': (3, 1.5),
            'speech_rate_consistency': (3, 1.5),
            'volume_fluctuation': (7, 1.5),
            'pause_patterns': (4, 1),
            'voice_tremor': (8, 1),
            'confidence_indicators': (3, 1),
            'stress_markers': (8, 1)
        },
        'Excited': {
            'energy_variation': (9, 1),
            'pitch_stability': (6, 1),
            'speech_rate_consistency': (6, 1),
            'volume_fluctuation': (9, 1),
            'pause_patterns': (5, 1),
            'voice_tremor': (4, 1),
            'confidence_indicators': (7, 1),
            'stress_markers': (4, 1)
        }
    }
    
    samples_per_emotion = num_samples // len(emotion_profiles)
    analyzer = SentimentAudioAnalyzer()
    
    for emotion, profile in emotion_profiles.items():
        logger.info(f"Generating {samples_per_emotion} samples for '{emotion}'...")
        
        for _ in range(samples_per_emotion):
            features = {}
            
            # Generate features for this emotion profile
            for feature_name, (mean, std) in profile.items():
                value = np.random.normal(mean, std)
                value = np.clip(value, 0, 10)  # Clamp to 0-10 range
                features[feature_name] = float(value)
            
            # Generate other features with reasonable defaults
            for feat in analyzer.sentiment_features:
                if feat not in features:
                    if 'stability' in feat or 'consistency' in feat:
                        features[feat] = float(np.random.normal(7, 1.5))
                    elif 'variation' in feat or 'fluctuation' in feat:
                        features[feat] = float(np.random.normal(5, 1.5))
                    elif 'tremor' in feat or 'jitter' in feat or 'shimmer' in feat:
                        features[feat] = float(np.random.normal(3, 1))
                    else:
                        features[feat] = float(np.random.normal(5, 1.5))
                    
                    features[feat] = np.clip(features[feat], 0, 10)
            
            # Calculate derived features
            features['emotional_intensity'] = np.mean([
                features.get('energy_variation', 5),
                features.get('volume_fluctuation', 5),
                features.get('spectral_centroid_variance', 5)
            ])
            
            features['confidence_indicators'] = np.mean([
                features.get('pitch_stability', 5),
                features.get('speech_rate_consistency', 5),
                features.get('formant_stability', 5)
            ])
            
            features['stress_markers'] = np.mean([
                10 - features.get('pitch_stability', 5),
                10 - features.get('speech_rate_consistency', 5),
                features.get('voice_tremor', 5)
            ])
            
            # Create feature vector
            feature_vector = [features.get(feat, 5.0) for feat in analyzer.sentiment_features]
            
            training_features.append(feature_vector)
            training_labels.append(emotion)
    
    return training_features, training_labels


def train_sentiment_model():
    """Main function to train the sentiment model"""
    logger.info("=" * 60)
    logger.info("Training Sentiment Audio Analyzer Model")
    logger.info("=" * 60)
    
    # Initialize analyzer
    analyzer = SentimentAudioAnalyzer()
    
    # Try to find existing audio files
    audio_files = find_audio_files("audio")
    
    training_features = []
    training_labels = []
    
    # Step 1: Use existing audio files if available
    if audio_files:
        logger.info(f"\nStep 1: Using {len(audio_files)} existing audio files...")
        features, labels = generate_training_data_from_audio(audio_files, analyzer)
        training_features.extend(features)
        training_labels.extend(labels)
        logger.info(f"  ✓ Extracted {len(features)} samples from audio files")
    
    # Step 2: Generate synthetic training data
    logger.info("\nStep 2: Generating synthetic training data...")
    synthetic_samples = max(500, 100 - len(training_features))  # At least 500 samples
    synth_features, synth_labels = generate_synthetic_training_data(synthetic_samples)
    training_features.extend(synth_features)
    training_labels.extend(synth_labels)
    logger.info(f"  ✓ Generated {len(synth_features)} synthetic samples")
    
    # Check if we have enough data
    if len(training_features) < 50:
        logger.error(f"Not enough training data! Only {len(training_features)} samples.")
        logger.error("Please add more audio files or increase synthetic samples.")
        return False
    
    logger.info(f"\nTotal training samples: {len(training_features)}")
    logger.info(f"Unique labels: {set(training_labels)}")
    
    # Step 3: Train the model
    logger.info("\nStep 3: Training the sentiment model...")
    try:
        analyzer.train_sentiment_model(training_features, training_labels)
        logger.info("  ✓ Model trained successfully!")
    except Exception as e:
        logger.error(f"  ✗ Error training model: {e}")
        return False
    
    # Step 4: Verify model was saved
    model_path = analyzer.model_path
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        logger.info(f"\n✓ Model saved successfully!")
        logger.info(f"  Location: {model_path}")
        logger.info(f"  Size: {file_size:.2f} MB")
        logger.info("\n" + "=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
        return True
    else:
        logger.error(f"Model file not found at {model_path}")
        return False


if __name__ == "__main__":
    success = train_sentiment_model()
    if success:
        print("\n✅ Sentiment model trained and saved successfully!")
        print("   The model is now available at: models/sentiment_model.pkl")
    else:
        print("\n❌ Failed to train sentiment model. Check logs for details.")
        exit(1)

