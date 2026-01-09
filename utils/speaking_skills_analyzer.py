"""
Speaking Skills Analyzer for AI Interview System
Analyzes interviewee speaking skills using machine learning models
"""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeakingSkillsAnalyzer:
    """
    Machine Learning model for analyzing speaking skills during interviews
    """
    
    def __init__(self, model_path: str = "models/speaking_skills_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            'speech_rate', 'pause_frequency', 'clarity_score', 'confidence_score',
            'energy_level', 'pitch_variation', 'volume_consistency', 'fluency_score',
            'articulation_score', 'pace_consistency', 'emphasis_effectiveness',
            'overall_communication_score'
        ]
        
        # Load pre-trained model if available
        self.load_model()
    
    def extract_audio_features(self, audio_file_path: str) -> Dict[str, float]:
        """
        Extract comprehensive audio features from interview recording
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_file_path, sr=None)
            
            features = {}
            
            # 1. Speech Rate (words per minute approximation)
            features['speech_rate'] = self._calculate_speech_rate(y, sr)
            
            # 2. Pause Frequency
            features['pause_frequency'] = self._calculate_pause_frequency(y, sr)
            
            # 3. Clarity Score (based on spectral centroid)
            features['clarity_score'] = self._calculate_clarity_score(y, sr)
            
            # 4. Confidence Score (based on energy and stability)
            features['confidence_score'] = self._calculate_confidence_score(y, sr)
            
            # 5. Energy Level
            features['energy_level'] = self._calculate_energy_level(y, sr)
            
            # 6. Pitch Variation
            features['pitch_variation'] = self._calculate_pitch_variation(y, sr)
            
            # 7. Volume Consistency
            features['volume_consistency'] = self._calculate_volume_consistency(y, sr)
            
            # 8. Fluency Score
            features['fluency_score'] = self._calculate_fluency_score(y, sr)
            
            # 9. Articulation Score
            features['articulation_score'] = self._calculate_articulation_score(y, sr)
            
            # 10. Pace Consistency
            features['pace_consistency'] = self._calculate_pace_consistency(y, sr)
            
            # 11. Emphasis Effectiveness
            features['emphasis_effectiveness'] = self._calculate_emphasis_effectiveness(y, sr)
            
            # 12. Overall Communication Score (weighted combination)
            features['overall_communication_score'] = self._calculate_overall_score(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return self._get_default_features()
    
    def _calculate_speech_rate(self, y: np.ndarray, sr: int) -> float:
        """Calculate approximate speech rate"""
        # Use onset detection to estimate speech segments
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        if len(onset_frames) > 1:
            duration = len(y) / sr
            speech_rate = len(onset_frames) / duration
            return min(speech_rate, 10.0)  # Normalize to 0-10 scale
        return 5.0
    
    def _calculate_pause_frequency(self, y: np.ndarray, sr: int) -> float:
        """Calculate frequency of pauses in speech"""
        # Detect silence segments
        silence_threshold = 0.01
        silence_mask = np.abs(y) < silence_threshold
        
        # Count silence segments
        silence_changes = np.diff(silence_mask.astype(int))
        silence_segments = np.sum(silence_changes == 1)
        
        duration = len(y) / sr
        pause_frequency = silence_segments / duration if duration > 0 else 0
        
        # Normalize to 0-10 scale (lower is better for pauses)
        return max(0, 10 - min(pause_frequency * 10, 10))
    
    def _calculate_clarity_score(self, y: np.ndarray, sr: int) -> float:
        """Calculate speech clarity based on spectral features"""
        # Spectral centroid (brightness of sound)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # MFCC features for clarity
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Calculate clarity score based on spectral properties
        clarity = np.mean(spectral_centroids) / 1000  # Normalize
        clarity = min(max(clarity, 0), 10)  # Clamp to 0-10
        
        return clarity
    
    def _calculate_confidence_score(self, y: np.ndarray, sr: int) -> float:
        """Calculate confidence based on energy stability and consistency"""
        # Energy envelope
        energy = librosa.feature.rms(y=y)[0]
        
        # Energy stability (lower variance = more confident)
        energy_stability = 1 / (1 + np.var(energy))
        
        # Energy consistency
        energy_consistency = np.std(energy) / np.mean(energy) if np.mean(energy) > 0 else 1
        energy_consistency = 1 / (1 + energy_consistency)
        
        # Combined confidence score
        confidence = (energy_stability + energy_consistency) / 2 * 10
        return min(max(confidence, 0), 10)
    
    def _calculate_energy_level(self, y: np.ndarray, sr: int) -> float:
        """Calculate overall energy level"""
        energy = librosa.feature.rms(y=y)[0]
        avg_energy = np.mean(energy)
        
        # Normalize to 0-10 scale
        energy_score = min(avg_energy * 100, 10)
        return max(energy_score, 0)
    
    def _calculate_pitch_variation(self, y: np.ndarray, sr: int) -> float:
        """Calculate pitch variation (monotone vs expressive)"""
        try:
            # Extract pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get the pitch values with highest magnitude
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 1:
                # Calculate pitch variation
                pitch_std = np.std(pitch_values)
                pitch_mean = np.mean(pitch_values)
                
                # Normalize variation (higher variation = more expressive)
                variation_score = min(pitch_std / pitch_mean * 10, 10) if pitch_mean > 0 else 5
                return max(variation_score, 0)
            
            return 5.0
        except:
            return 5.0
    
    def _calculate_volume_consistency(self, y: np.ndarray, sr: int) -> float:
        """Calculate volume consistency throughout speech"""
        # RMS energy over time
        energy = librosa.feature.rms(y=y)[0]
        
        # Calculate consistency (lower variance = more consistent)
        energy_std = np.std(energy)
        energy_mean = np.mean(energy)
        
        if energy_mean > 0:
            consistency = 1 / (1 + energy_std / energy_mean)
            return consistency * 10
        return 5.0
    
    def _calculate_fluency_score(self, y: np.ndarray, sr: int) -> float:
        """Calculate speech fluency (smoothness)"""
        # Detect speech segments
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        
        if len(onset_frames) > 1:
            # Calculate intervals between speech segments
            intervals = np.diff(onset_frames)
            
            # Fluency is inversely related to irregular intervals
            interval_std = np.std(intervals)
            interval_mean = np.mean(intervals)
            
            if interval_mean > 0:
                fluency = 1 / (1 + interval_std / interval_mean)
                return fluency * 10
        
        return 5.0
    
    def _calculate_articulation_score(self, y: np.ndarray, sr: int) -> float:
        """Calculate articulation clarity"""
        # Use spectral contrast for articulation
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Higher contrast = better articulation
        contrast_mean = np.mean(spectral_contrast)
        
        # Normalize to 0-10 scale
        articulation = min(contrast_mean / 10, 10)
        return max(articulation, 0)
    
    def _calculate_pace_consistency(self, y: np.ndarray, sr: int) -> float:
        """Calculate consistency of speaking pace"""
        # Use tempo estimation
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Calculate beat intervals
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            interval_std = np.std(beat_intervals)
            interval_mean = np.mean(beat_intervals)
            
            if interval_mean > 0:
                consistency = 1 / (1 + interval_std / interval_mean)
                return consistency * 10
        
        return 5.0
    
    def _calculate_emphasis_effectiveness(self, y: np.ndarray, sr: int) -> float:
        """Calculate effectiveness of emphasis and stress"""
        # Energy variations for emphasis
        energy = librosa.feature.rms(y=y)[0]
        
        # Look for energy peaks (emphasis)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(energy, height=np.mean(energy) * 1.2)
        
        # More peaks = more emphasis
        emphasis_score = min(len(peaks) / 10, 10)
        return max(emphasis_score, 0)
    
    def _calculate_overall_score(self, features: Dict[str, float]) -> float:
        """Calculate weighted overall communication score"""
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
        
        return min(overall_score, 10)
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features if extraction fails"""
        return {feature: 5.0 for feature in self.feature_names}
    
    def analyze_speaking_skills(self, audio_file_path: str) -> Dict[str, any]:
        """
        Analyze speaking skills and provide comprehensive feedback
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary containing analysis results and feedback
        """
        try:
            # Extract features
            features = self.extract_audio_features(audio_file_path)
            
            # Get speaking skills assessment
            assessment = self._assess_speaking_skills(features)
            
            # Generate detailed feedback
            feedback = self._generate_feedback(features, assessment)
            
            # Calculate overall score
            overall_score = features['overall_communication_score']
            
            return {
                'features': features,
                'assessment': assessment,
                'feedback': feedback,
                'overall_score': overall_score,
                'score_breakdown': self._get_score_breakdown(features)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing speaking skills: {e}")
            return {
                'error': str(e),
                'features': self._get_default_features(),
                'assessment': 'Unable to assess',
                'feedback': 'Analysis failed due to technical error',
                'overall_score': 0.0
            }
    
    def _assess_speaking_skills(self, features: Dict[str, float]) -> str:
        """Assess speaking skills based on extracted features"""
        overall_score = features['overall_communication_score']
        
        if overall_score >= 8.5:
            return "Exceptional"
        elif overall_score >= 7.0:
            return "Excellent"
        elif overall_score >= 5.5:
            return "Good"
        elif overall_score >= 4.0:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _generate_feedback(self, features: Dict[str, float], assessment: str) -> Dict[str, str]:
        """Generate detailed feedback for each speaking skill area"""
        feedback = {}
        
        # Speech Rate Feedback
        if features['speech_rate'] < 3:
            feedback['speech_rate'] = "Your speech rate is too slow. Try to maintain a moderate pace."
        elif features['speech_rate'] > 8:
            feedback['speech_rate'] = "Your speech rate is too fast. Slow down for better clarity."
        else:
            feedback['speech_rate'] = "Good speech rate. Maintain this pace."
        
        # Clarity Feedback
        if features['clarity_score'] < 5:
            feedback['clarity_score'] = "Work on pronunciation and articulation. Practice clear speech."
        else:
            feedback['clarity_score'] = "Good clarity. Your words are well-articulated."
        
        # Confidence Feedback
        if features['confidence_score'] < 5:
            feedback['confidence_score'] = "Build confidence through practice. Maintain steady energy levels."
        else:
            feedback['confidence_score'] = "Excellent confidence. Your voice projects well."
        
        # Fluency Feedback
        if features['fluency_score'] < 5:
            feedback['fluency_score'] = "Practice smooth transitions between thoughts. Reduce filler words."
        else:
            feedback['fluency_score'] = "Good fluency. Your speech flows naturally."
        
        # Overall Feedback
        if assessment == "Exceptional":
            feedback['overall'] = "Outstanding speaking skills! You demonstrate excellent communication abilities."
        elif assessment == "Excellent":
            feedback['overall'] = "Very strong speaking skills with room for minor improvements."
        elif assessment == "Good":
            feedback['overall'] = "Solid foundation. Focus on the areas mentioned above for improvement."
        elif assessment == "Fair":
            feedback['overall'] = "Basic speaking skills present. Dedicated practice will help significantly."
        else:
            feedback['overall'] = "Fundamental speaking skills need development. Consider speech coaching."
        
        return feedback
    
    def _get_score_breakdown(self, features: Dict[str, float]) -> Dict[str, Dict[str, any]]:
        """Get detailed breakdown of each speaking skill score"""
        breakdown = {}
        
        for feature, score in features.items():
            if feature != 'overall_communication_score':
                breakdown[feature] = {
                    'score': score,
                    'max_score': 10.0,
                    'percentage': (score / 10.0) * 100,
                    'grade': self._get_grade(score)
                }
        
        return breakdown
    
    def _get_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 9.0:
            return "A+"
        elif score >= 8.0:
            return "A"
        elif score >= 7.0:
            return "B+"
        elif score >= 6.0:
            return "B"
        elif score >= 5.0:
            return "C+"
        elif score >= 4.0:
            return "C"
        elif score >= 3.0:
            return "D"
        else:
            return "F"
    
    def train_model(self, training_data: List[Dict], labels: List[str]) -> None:
        """
        Train the speaking skills assessment model
        
        Args:
            training_data: List of feature dictionaries
            labels: List of assessment labels
        """
        try:
            # Prepare training data
            X = pd.DataFrame(training_data)
            y = self.label_encoder.fit_transform(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model (using Random Forest for interpretability)
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model trained successfully. Accuracy: {accuracy:.2f}")
            
            # Save model
            self.save_model()
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
    
    def predict_assessment(self, features: Dict[str, float]) -> str:
        """Predict speaking skills assessment using trained model"""
        if self.model is None:
            return self._assess_speaking_skills(features)
        
        try:
            # Prepare features
            feature_vector = np.array([features[feature] for feature in self.feature_names])
            feature_vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
            
            # Make prediction
            prediction = self.model.predict(feature_vector_scaled)[0]
            assessment = self.label_encoder.inverse_transform([prediction])[0]
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return self._assess_speaking_skills(features)
    
    def save_model(self) -> None:
        """Save the trained model and scaler"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }
            
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self) -> None:
        """Load pre-trained model and scaler"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.label_encoder = model_data['label_encoder']
                self.feature_names = model_data['feature_names']
                
                logger.info(f"Model loaded from {self.model_path}")
            else:
                logger.info("No pre-trained model found. Using rule-based assessment.")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Using rule-based assessment as fallback.")


# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SpeakingSkillsAnalyzer()
    
    # Example audio file path (replace with actual path)
    audio_file = "path/to/audio/file.wav"
    
    if os.path.exists(audio_file):
        # Analyze speaking skills
        results = analyzer.analyze_speaking_skills(audio_file)
        
        print("Speaking Skills Analysis Results:")
        print(f"Overall Score: {results['overall_score']:.2f}/10")
        print(f"Assessment: {results['assessment']}")
        print("\nDetailed Feedback:")
        for area, feedback in results['feedback'].items():
            print(f"{area}: {feedback}")
        
        print("\nScore Breakdown:")
        for skill, details in results['score_breakdown'].items():
            print(f"{skill}: {details['score']:.2f}/10 ({details['grade']})")
    else:
        print(f"Audio file not found: {audio_file}")
