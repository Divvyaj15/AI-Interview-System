"""
Sentimental Audio Analyzer for AI Interview System
Analyzes emotional states, confidence, stress levels, and emotional intelligence
"""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from typing import Dict, List, Tuple, Optional
import logging
from scipy import signal
from scipy.stats import skew, kurtosis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAudioAnalyzer:
    """
    Advanced sentiment analysis for interview audio recordings
    Detects emotional states, confidence, stress, and emotional intelligence
    """
    
    def __init__(self, model_path: str = "models/sentiment_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Emotional states to detect
        self.emotion_labels = [
            'Confident', 'Anxious', 'Enthusiastic', 'Calm', 
            'Stressed', 'Excited', 'Nervous', 'Professional'
        ]
        
        # Sentiment features
        self.sentiment_features = [
            'energy_variation', 'pitch_stability', 'speech_rate_consistency',
            'volume_fluctuation', 'pause_patterns', 'voice_tremor',
            'spectral_centroid_variance', 'mfcc_variation', 'harmonic_ratio',
            'noise_level', 'formant_stability', 'jitter', 'shimmer',
            'emotional_intensity', 'confidence_indicators', 'stress_markers'
        ]
        
        # Load pre-trained model if available
        self.load_model()
    
    def analyze_sentiment(self, audio_file_path: str) -> Dict[str, any]:
        """
        Comprehensive sentiment analysis of interview audio
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            # Extract sentiment features
            features = self.extract_sentiment_features(audio_file_path)
            
            # Analyze emotional states
            emotional_analysis = self._analyze_emotional_states(features)
            
            # Detect confidence indicators
            confidence_analysis = self._analyze_confidence(features)
            
            # Identify stress markers
            stress_analysis = self._analyze_stress_indicators(features)
            
            # Calculate emotional intelligence score
            emotional_intelligence = self._calculate_emotional_intelligence(features)
            
            # Generate overall sentiment assessment
            overall_sentiment = self._assess_overall_sentiment(features)
            
            return {
                'features': features,
                'emotional_states': emotional_analysis,
                'confidence_analysis': confidence_analysis,
                'stress_analysis': stress_analysis,
                'emotional_intelligence': emotional_intelligence,
                'overall_sentiment': overall_sentiment,
                'recommendations': self._generate_sentiment_recommendations(features)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'error': str(e),
                'features': self._get_default_sentiment_features(),
                'emotional_states': {'primary': 'Neutral', 'confidence': 0.5},
                'confidence_analysis': {'level': 'Medium', 'indicators': []},
                'stress_analysis': {'level': 'Low', 'markers': []},
                'emotional_intelligence': 5.0,
                'overall_sentiment': 'Neutral',
                'recommendations': ['Analysis failed due to technical error']
            }
    
    def extract_sentiment_features(self, audio_file_path: str) -> Dict[str, float]:
        """
        Extract comprehensive sentiment features from audio
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary of sentiment features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_file_path, sr=None)
            
            features = {}
            
            # 1. Energy Variation (emotional expressiveness)
            features['energy_variation'] = self._calculate_energy_variation(y, sr)
            
            # 2. Pitch Stability (confidence indicator)
            features['pitch_stability'] = self._calculate_pitch_stability(y, sr)
            
            # 3. Speech Rate Consistency (nervousness indicator)
            features['speech_rate_consistency'] = self._calculate_speech_rate_consistency(y, sr)
            
            # 4. Volume Fluctuation (emotional intensity)
            features['volume_fluctuation'] = self._calculate_volume_fluctuation(y, sr)
            
            # 5. Pause Patterns (thought organization)
            features['pause_patterns'] = self._calculate_pause_patterns(y, sr)
            
            # 6. Voice Tremor (stress indicator)
            features['voice_tremor'] = self._calculate_voice_tremor(y, sr)
            
            # 7. Spectral Centroid Variance (emotional variation)
            features['spectral_centroid_variance'] = self._calculate_spectral_centroid_variance(y, sr)
            
            # 8. MFCC Variation (speech characteristics)
            features['mfcc_variation'] = self._calculate_mfcc_variation(y, sr)
            
            # 9. Harmonic Ratio (voice quality)
            features['harmonic_ratio'] = self._calculate_harmonic_ratio(y, sr)
            
            # 10. Noise Level (recording quality)
            features['noise_level'] = self._calculate_noise_level(y, sr)
            
            # 11. Formant Stability (articulation)
            features['formant_stability'] = self._calculate_formant_stability(y, sr)
            
            # 12. Jitter (voice roughness)
            features['jitter'] = self._calculate_jitter(y, sr)
            
            # 13. Shimmer (amplitude variation)
            features['shimmer'] = self._calculate_shimmer(y, sr)
            
            # 14. Emotional Intensity (overall emotional expression)
            features['emotional_intensity'] = self._calculate_emotional_intensity(features)
            
            # 15. Confidence Indicators (combined confidence metrics)
            features['confidence_indicators'] = self._calculate_confidence_indicators(features)
            
            # 16. Stress Markers (combined stress metrics)
            features['stress_markers'] = self._calculate_stress_markers(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting sentiment features: {e}")
            return self._get_default_sentiment_features()
    
    def _calculate_energy_variation(self, y: np.ndarray, sr: int) -> float:
        """Calculate energy variation as emotional expressiveness indicator"""
        # RMS energy over time
        energy = librosa.feature.rms(y=y)[0]
        
        # Calculate coefficient of variation
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        
        if energy_mean > 0:
            cv = energy_std / energy_mean
            # Normalize to 0-10 scale (higher = more expressive)
            return min(cv * 10, 10)
        return 5.0
    
    def _calculate_pitch_stability(self, y: np.ndarray, sr: int) -> float:
        """Calculate pitch stability as confidence indicator"""
        try:
            # Extract pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get pitch values with highest magnitude
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 1:
                # Calculate pitch stability (lower variance = more stable = more confident)
                pitch_std = np.std(pitch_values)
                pitch_mean = np.mean(pitch_values)
                
                if pitch_mean > 0:
                    stability = 1 / (1 + pitch_std / pitch_mean)
                    return stability * 10
            
            return 5.0
        except:
            return 5.0
    
    def _calculate_speech_rate_consistency(self, y: np.ndarray, sr: int) -> float:
        """Calculate speech rate consistency (nervousness indicator)"""
        # Detect speech segments
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        
        if len(onset_frames) > 2:
            # Calculate intervals between speech segments
            intervals = np.diff(onset_frames)
            
            # Consistency is inversely related to standard deviation
            interval_std = np.std(intervals)
            interval_mean = np.mean(intervals)
            
            if interval_mean > 0:
                consistency = 1 / (1 + interval_std / interval_mean)
                return consistency * 10
        
        return 5.0
    
    def _calculate_volume_fluctuation(self, y: np.ndarray, sr: int) -> float:
        """Calculate volume fluctuation as emotional intensity"""
        # RMS energy over time
        energy = librosa.feature.rms(y=y)[0]
        
        # Calculate volume changes
        volume_changes = np.diff(energy)
        
        # Average absolute change
        avg_change = np.mean(np.abs(volume_changes))
        max_change = np.max(energy)
        
        if max_change > 0:
            fluctuation = avg_change / max_change
            return min(fluctuation * 20, 10)  # Normalize to 0-10
        
        return 5.0
    
    def _calculate_pause_patterns(self, y: np.ndarray, sr: int) -> float:
        """Calculate pause patterns for thought organization"""
        # Detect silence segments
        silence_threshold = 0.01
        silence_mask = np.abs(y) < silence_threshold
        
        # Find silence segments
        silence_starts = np.where(np.diff(silence_mask.astype(int)) == 1)[0]
        silence_ends = np.where(np.diff(silence_mask.astype(int)) == -1)[0]
        
        if len(silence_starts) > 0 and len(silence_ends) > 0:
            # Calculate silence durations
            silence_durations = []
            for start, end in zip(silence_starts, silence_ends):
                if end > start:
                    duration = (end - start) / sr
                    if 0.1 < duration < 3.0:  # Reasonable pause range
                        silence_durations.append(duration)
            
            if silence_durations:
                # Calculate pause consistency
                pause_std = np.std(silence_durations)
                pause_mean = np.mean(silence_durations)
                
                if pause_mean > 0:
                    consistency = 1 / (1 + pause_std / pause_mean)
                    return consistency * 10
        
        return 5.0
    
    def _calculate_voice_tremor(self, y: np.ndarray, sr: int) -> float:
        """Calculate voice tremor as stress indicator"""
        try:
            # Extract pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get pitch values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 1:
                # Calculate pitch variation (tremor)
                pitch_std = np.std(pitch_values)
                pitch_mean = np.mean(pitch_values)
                
                if pitch_mean > 0:
                    tremor = pitch_std / pitch_mean
                    # Normalize to 0-10 scale (higher = more tremor = more stress)
                    return min(tremor * 10, 10)
            
            return 5.0
        except:
            return 5.0
    
    def _calculate_spectral_centroid_variance(self, y: np.ndarray, sr: int) -> float:
        """Calculate spectral centroid variance for emotional variation"""
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Calculate variance
        centroid_variance = np.var(spectral_centroids)
        
        # Normalize to 0-10 scale
        normalized_variance = min(centroid_variance / 1000000, 10)
        return max(normalized_variance, 0)
    
    def _calculate_mfcc_variation(self, y: np.ndarray, sr: int) -> float:
        """Calculate MFCC variation for speech characteristics"""
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Calculate variation across time
        mfcc_variance = np.var(mfccs, axis=1)
        
        # Average variance
        avg_variance = np.mean(mfcc_variance)
        
        # Normalize to 0-10 scale
        normalized_variance = min(avg_variance / 10, 10)
        return max(normalized_variance, 0)
    
    def _calculate_harmonic_ratio(self, y: np.ndarray, sr: int) -> float:
        """Calculate harmonic ratio for voice quality"""
        try:
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Calculate energy ratio
            harmonic_energy = np.sum(y_harmonic ** 2)
            total_energy = np.sum(y ** 2)
            
            if total_energy > 0:
                harmonic_ratio = harmonic_energy / total_energy
                return harmonic_ratio * 10
            
            return 5.0
        except:
            return 5.0
    
    def _calculate_noise_level(self, y: np.ndarray, sr: int) -> float:
        """Calculate noise level for recording quality"""
        # High-frequency noise detection
        # Apply high-pass filter
        from scipy.signal import butter, filtfilt
        
        try:
            # Design high-pass filter
            nyquist = sr / 2
            cutoff = 8000  # 8 kHz
            normal_cutoff = cutoff / nyquist
            b, a = butter(4, normal_cutoff, btype='high', analog=False)
            
            # Apply filter
            filtered_signal = filtfilt(b, a, y)
            
            # Calculate noise level
            noise_energy = np.sum(filtered_signal ** 2)
            total_energy = np.sum(y ** 2)
            
            if total_energy > 0:
                noise_ratio = noise_energy / total_energy
                # Lower noise ratio is better, so invert
                noise_score = 10 * (1 - noise_ratio)
                return max(noise_score, 0)
            
            return 5.0
        except:
            return 5.0
    
    def _calculate_formant_stability(self, y: np.ndarray, sr: int) -> float:
        """Calculate formant stability for articulation"""
        try:
            # Extract formants using LPC
            from scipy.signal import lpc
            
            # Frame the signal
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            formant_frequencies = []
            
            for i in range(0, len(y) - frame_length, hop_length):
                frame = y[i:i + frame_length]
                
                # Apply window
                windowed_frame = frame * np.hanning(len(frame))
                
                # LPC analysis
                try:
                    a = lpc(windowed_frame, 12)[0]
                    
                    # Find roots of LPC polynomial
                    roots = np.roots(a)
                    
                    # Convert to frequencies
                    angles = np.angle(roots)
                    freqs = angles * sr / (2 * np.pi)
                    
                    # Keep only positive frequencies
                    freqs = freqs[freqs > 0]
                    
                    if len(freqs) > 0:
                        # Sort by frequency and take first few formants
                        freqs = np.sort(freqs)
                        formant_frequencies.extend(freqs[:3])
                
                except:
                    continue
            
            if len(formant_frequencies) > 1:
                # Calculate stability
                formant_std = np.std(formant_frequencies)
                formant_mean = np.mean(formant_frequencies)
                
                if formant_mean > 0:
                    stability = 1 / (1 + formant_std / formant_mean)
                    return stability * 10
            
            return 5.0
        except:
            return 5.0
    
    def _calculate_jitter(self, y: np.ndarray, sr: int) -> float:
        """Calculate jitter (pitch period variation)"""
        try:
            # Extract pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get pitch values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 1:
                # Calculate jitter (relative average perturbation)
                pitch_periods = 1 / np.array(pitch_values)
                
                # Calculate jitter
                jitter_values = np.abs(np.diff(pitch_periods))
                avg_jitter = np.mean(jitter_values)
                avg_period = np.mean(pitch_periods)
                
                if avg_period > 0:
                    relative_jitter = avg_jitter / avg_period
                    # Normalize to 0-10 scale (lower is better)
                    jitter_score = 10 * (1 - min(relative_jitter, 1))
                    return max(jitter_score, 0)
            
            return 5.0
        except:
            return 5.0
    
    def _calculate_shimmer(self, y: np.ndarray, sr: int) -> float:
        """Calculate shimmer (amplitude variation)"""
        try:
            # RMS energy over time
            energy = librosa.feature.rms(y=y)[0]
            
            # Calculate shimmer
            energy_diff = np.abs(np.diff(energy))
            avg_energy_diff = np.mean(energy_diff)
            avg_energy = np.mean(energy)
            
            if avg_energy > 0:
                relative_shimmer = avg_energy_diff / avg_energy
                # Normalize to 0-10 scale (lower is better)
                shimmer_score = 10 * (1 - min(relative_shimmer, 1))
                return max(shimmer_score, 0)
            
            return 5.0
        except:
            return 5.0
    
    def _calculate_emotional_intensity(self, features: Dict[str, float]) -> float:
        """Calculate overall emotional intensity"""
        # Combine relevant features
        intensity_factors = [
            features.get('energy_variation', 5),
            features.get('volume_fluctuation', 5),
            features.get('spectral_centroid_variance', 5),
            features.get('mfcc_variation', 5)
        ]
        
        return np.mean(intensity_factors)
    
    def _calculate_confidence_indicators(self, features: Dict[str, float]) -> float:
        """Calculate combined confidence indicators"""
        # Combine confidence-related features
        confidence_factors = [
            features.get('pitch_stability', 5),
            features.get('speech_rate_consistency', 5),
            features.get('formant_stability', 5),
            features.get('jitter', 5),
            features.get('shimmer', 5)
        ]
        
        return np.mean(confidence_factors)
    
    def _calculate_stress_markers(self, features: Dict[str, float]) -> float:
        """Calculate combined stress markers"""
        # Combine stress-related features
        stress_factors = [
            10 - features.get('pitch_stability', 5),  # Invert stability
            10 - features.get('speech_rate_consistency', 5),  # Invert consistency
            features.get('voice_tremor', 5),
            features.get('jitter', 5),
            features.get('shimmer', 5)
        ]
        
        return np.mean(stress_factors)
    
    def _analyze_emotional_states(self, features: Dict[str, float]) -> Dict[str, any]:
        """Analyze emotional states based on features"""
        # Use rule-based analysis for emotional state detection
        emotional_scores = {}
        
        # Confidence indicators
        confidence_score = features.get('confidence_indicators', 5)
        if confidence_score >= 8:
            emotional_scores['Confident'] = 0.9
        elif confidence_score >= 6:
            emotional_scores['Confident'] = 0.7
        
        # Stress indicators
        stress_score = features.get('stress_markers', 5)
        if stress_score >= 8:
            emotional_scores['Stressed'] = 0.9
        elif stress_score >= 6:
            emotional_scores['Stressed'] = 0.7
        
        # Enthusiasm indicators
        enthusiasm_score = features.get('emotional_intensity', 5)
        if enthusiasm_score >= 8:
            emotional_scores['Enthusiastic'] = 0.9
        elif enthusiasm_score >= 6:
            emotional_scores['Enthusiastic'] = 0.7
        
        # Calm indicators
        calm_score = (features.get('pitch_stability', 5) + 
                     features.get('speech_rate_consistency', 5)) / 2
        if calm_score >= 8:
            emotional_scores['Calm'] = 0.9
        elif calm_score >= 6:
            emotional_scores['Calm'] = 0.7
        
        # Professional indicators
        professional_score = (features.get('formant_stability', 5) + 
                           features.get('harmonic_ratio', 5)) / 2
        if professional_score >= 8:
            emotional_scores['Professional'] = 0.9
        elif professional_score >= 6:
            emotional_scores['Professional'] = 0.7
        
        # Find primary emotion
        if emotional_scores:
            primary_emotion = max(emotional_scores.items(), key=lambda x: x[1])
        else:
            primary_emotion = ('Neutral', 0.5)
        
        return {
            'primary': primary_emotion[0],
            'confidence': primary_emotion[1],
            'all_emotions': emotional_scores
        }
    
    def _analyze_confidence(self, features: Dict[str, float]) -> Dict[str, any]:
        """Analyze confidence level and indicators"""
        confidence_score = features.get('confidence_indicators', 5)
        
        if confidence_score >= 8:
            level = "High"
            indicators = ["Stable pitch", "Consistent speech rate", "Clear articulation"]
        elif confidence_score >= 6:
            level = "Medium"
            indicators = ["Moderate pitch stability", "Some speech rate variation"]
        else:
            level = "Low"
            indicators = ["Pitch instability", "Variable speech rate", "Voice tremor"]
        
        return {
            'level': level,
            'score': confidence_score,
            'indicators': indicators
        }
    
    def _analyze_stress_indicators(self, features: Dict[str, float]) -> Dict[str, any]:
        """Analyze stress level and markers"""
        stress_score = features.get('stress_markers', 5)
        
        if stress_score >= 8:
            level = "High"
            markers = ["Voice tremor", "Pitch instability", "Irregular speech patterns"]
        elif stress_score >= 6:
            level = "Medium"
            markers = ["Some pitch variation", "Moderate speech inconsistency"]
        else:
            level = "Low"
            markers = ["Stable voice", "Consistent speech patterns"]
        
        return {
            'level': level,
            'score': stress_score,
            'markers': markers
        }
    
    def _calculate_emotional_intelligence(self, features: Dict[str, float]) -> float:
        """Calculate emotional intelligence score"""
        # Combine multiple factors for emotional intelligence
        ei_factors = [
            features.get('confidence_indicators', 5),
            features.get('formant_stability', 5),
            features.get('harmonic_ratio', 5),
            features.get('pause_patterns', 5),
            features.get('emotional_intensity', 5)
        ]
        
        # Calculate weighted average
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]
        ei_score = sum(factor * weight for factor, weight in zip(ei_factors, weights))
        
        return min(ei_score, 10)
    
    def _assess_overall_sentiment(self, features: Dict[str, float]) -> str:
        """Assess overall sentiment"""
        # Combine multiple sentiment indicators
        sentiment_score = (
            features.get('confidence_indicators', 5) * 0.3 +
            features.get('emotional_intensity', 5) * 0.2 +
            (10 - features.get('stress_markers', 5)) * 0.3 +
            features.get('harmonic_ratio', 5) * 0.2
        )
        
        if sentiment_score >= 8:
            return "Positive"
        elif sentiment_score >= 6:
            return "Neutral"
        else:
            return "Negative"
    
    def _generate_sentiment_recommendations(self, features: Dict[str, float]) -> List[str]:
        """Generate recommendations based on sentiment analysis"""
        recommendations = []
        
        # Confidence recommendations
        confidence_score = features.get('confidence_indicators', 5)
        if confidence_score < 6:
            recommendations.append("Practice deep breathing exercises before interviews")
            recommendations.append("Work on maintaining steady voice projection")
            recommendations.append("Consider speech coaching for confidence building")
        
        # Stress recommendations
        stress_score = features.get('stress_markers', 5)
        if stress_score > 6:
            recommendations.append("Practice stress management techniques")
            recommendations.append("Take short pauses to collect thoughts")
            recommendations.append("Consider meditation or relaxation exercises")
        
        # Emotional expression recommendations
        emotional_score = features.get('emotional_intensity', 5)
        if emotional_score < 5:
            recommendations.append("Work on expressing enthusiasm and engagement")
            recommendations.append("Practice varying your vocal tone")
            recommendations.append("Show more emotional connection to your responses")
        
        # Voice quality recommendations
        voice_score = (features.get('harmonic_ratio', 5) + 
                      features.get('formant_stability', 5)) / 2
        if voice_score < 6:
            recommendations.append("Practice clear articulation and pronunciation")
            recommendations.append("Work on voice projection and clarity")
            recommendations.append("Consider vocal exercises for better quality")
        
        if not recommendations:
            recommendations.append("Excellent emotional control and expression")
            recommendations.append("Maintain your current communication style")
        
        return recommendations
    
    def _get_default_sentiment_features(self) -> Dict[str, float]:
        """Return default features if extraction fails"""
        return {feature: 5.0 for feature in self.sentiment_features}
    
    def train_sentiment_model(self, training_data: List[Dict], labels: List[str]) -> None:
        """Train the sentiment analysis model"""
        try:
            # Prepare training data
            X = pd.DataFrame(training_data)
            y = self.label_encoder.fit_transform(labels)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            
            # Save model
            self.save_model()
            
            logger.info("Sentiment model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training sentiment model: {e}")
    
    def predict_emotion(self, features: Dict[str, float]) -> str:
        """Predict emotional state using trained model"""
        if self.model is None:
            return self._analyze_emotional_states(features)['primary']
        
        try:
            # Prepare features
            feature_vector = np.array([features[feature] for feature in self.sentiment_features])
            feature_vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
            
            # Make prediction
            prediction = self.model.predict(feature_vector_scaled)[0]
            emotion = self.label_encoder.inverse_transform([prediction])[0]
            
            return emotion
            
        except Exception as e:
            logger.error(f"Error making emotion prediction: {e}")
            return self._analyze_emotional_states(features)['primary']
    
    def save_model(self) -> None:
        """Save the trained model and scaler"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'sentiment_features': self.sentiment_features
            }
            
            joblib.dump(model_data, self.model_path)
            logger.info(f"Sentiment model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving sentiment model: {e}")
    
    def load_model(self) -> None:
        """Load pre-trained sentiment model"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.label_encoder = model_data['label_encoder']
                self.sentiment_features = model_data['sentiment_features']
                
                logger.info(f"Sentiment model loaded from {self.model_path}")
            else:
                logger.info("No pre-trained sentiment model found. Using rule-based analysis.")
                
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            logger.info("Using rule-based sentiment analysis as fallback.")


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SentimentAudioAnalyzer()
    
    # Example audio file path
    audio_file = "path/to/audio/file.wav"
    
    if os.path.exists(audio_file):
        # Analyze sentiment
        results = analyzer.analyze_sentiment(audio_file)
        
        print("Sentiment Analysis Results:")
        print(f"Primary Emotion: {results['emotional_states']['primary']}")
        print(f"Confidence Level: {results['confidence_analysis']['level']}")
        print(f"Stress Level: {results['stress_analysis']['level']}")
        print(f"Emotional Intelligence: {results['emotional_intelligence']:.2f}/10")
        print(f"Overall Sentiment: {results['overall_sentiment']}")
        
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"- {rec}")
    else:
        print(f"Audio file not found: {audio_file}")
