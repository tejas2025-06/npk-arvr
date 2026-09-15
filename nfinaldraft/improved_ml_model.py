import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import os

class ImprovedNPKAnalyzer:
    def __init__(self):
        self.model_path = 'improved_npk_model.pkl'
        self.dataset_path = 'Crop_recommendation.csv'
        self.model = None
        self.scaler = None
        
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            print("Improved model not found. Training with advanced techniques...")
            self._train_improved_model()
    
    def _create_advanced_features(self, df):
        """Create advanced features from NPK values"""
        # Start with original features
        enhanced_df = df[['N', 'P', 'K', 'label']].copy()
        
        # Ratios - these are very important for crop selection
        enhanced_df['N_P_ratio'] = enhanced_df['N'] / (enhanced_df['P'] + 0.1)
        enhanced_df['N_K_ratio'] = enhanced_df['N'] / (enhanced_df['K'] + 0.1)
        enhanced_df['P_K_ratio'] = enhanced_df['P'] / (enhanced_df['K'] + 0.1)
        
        # Inverse ratios
        enhanced_df['P_N_ratio'] = enhanced_df['P'] / (enhanced_df['N'] + 0.1)
        enhanced_df['K_N_ratio'] = enhanced_df['K'] / (enhanced_df['N'] + 0.1)
        enhanced_df['K_P_ratio'] = enhanced_df['K'] / (enhanced_df['P'] + 0.1)
        
        # Total and averages
        enhanced_df['NPK_total'] = enhanced_df['N'] + enhanced_df['P'] + enhanced_df['K']
        enhanced_df['NPK_avg'] = enhanced_df['NPK_total'] / 3
        
        # Proportions (normalized to sum to 1)
        enhanced_df['N_prop'] = enhanced_df['N'] / enhanced_df['NPK_total']
        enhanced_df['P_prop'] = enhanced_df['P'] / enhanced_df['NPK_total']
        enhanced_df['K_prop'] = enhanced_df['K'] / enhanced_df['NPK_total']
        
        # Squared terms
        enhanced_df['N_sq'] = enhanced_df['N'] ** 2
        enhanced_df['P_sq'] = enhanced_df['P'] ** 2
        enhanced_df['K_sq'] = enhanced_df['K'] ** 2
        
        # Square root terms
        enhanced_df['N_sqrt'] = np.sqrt(enhanced_df['N'])
        enhanced_df['P_sqrt'] = np.sqrt(enhanced_df['P'])
        enhanced_df['K_sqrt'] = np.sqrt(enhanced_df['K'])
        
        # Interaction terms
        enhanced_df['NP_product'] = enhanced_df['N'] * enhanced_df['P']
        enhanced_df['NK_product'] = enhanced_df['N'] * enhanced_df['K']
        enhanced_df['PK_product'] = enhanced_df['P'] * enhanced_df['K']
        
        # Nutrient balance indicators
        enhanced_df['max_nutrient'] = enhanced_df[['N', 'P', 'K']].max(axis=1)
        enhanced_df['min_nutrient'] = enhanced_df[['N', 'P', 'K']].min(axis=1)
        enhanced_df['nutrient_range'] = enhanced_df['max_nutrient'] - enhanced_df['min_nutrient']
        enhanced_df['nutrient_std'] = enhanced_df[['N', 'P', 'K']].std(axis=1)
        
        # Log transformations (for skewed distributions)
        enhanced_df['N_log'] = np.log1p(enhanced_df['N'])
        enhanced_df['P_log'] = np.log1p(enhanced_df['P'])
        enhanced_df['K_log'] = np.log1p(enhanced_df['K'])
        
        return enhanced_df
    
    def _train_improved_model(self):
        try:
            # Load data
            df = pd.read_csv(self.dataset_path)
            print(f"Original dataset shape: {df.shape}")
            
            # Create advanced features
            enhanced_df = self._create_advanced_features(df)
            print(f"Enhanced dataset shape: {enhanced_df.shape}")
            
            # Prepare features and target
            feature_cols = [col for col in enhanced_df.columns if col != 'label']
            X = enhanced_df[feature_cols]
            y = enhanced_df['label']
            
            print(f"Number of features: {len(feature_cols)}")
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            print(f"Training samples: {X_train.shape[0]}")
            print(f"Testing samples: {X_test.shape[0]}")
            
            # Train optimized Random Forest
            self.model = RandomForestClassifier(
                n_estimators=1200,           # More trees
                max_depth=30,                # Deeper trees
                min_samples_split=3,         # Balanced splitting
                min_samples_leaf=1,          # Allow fine-grained splits
                max_features='sqrt',         # Good balance
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',     # Handle class imbalance
                oob_score=True              # Out-of-bag scoring
            )
            
            print("Training improved Random Forest model...")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_accuracy = self.model.score(X_train_scaled, y_train)
            test_accuracy = self.model.score(X_test_scaled, y_test)
            oob_accuracy = self.model.oob_score_
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            
            print(f"\\n=== IMPROVED MODEL PERFORMANCE ===")
            print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
            print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"OOB Accuracy: {oob_accuracy:.4f} ({oob_accuracy*100:.2f}%)")
            print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"Overfitting: {(train_accuracy-test_accuracy)*100:.2f}%")
            
            # Feature importance analysis
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\\n=== TOP 15 MOST IMPORTANT FEATURES ===")
            for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<15}: {row['importance']:.4f}")
            
            # Save model
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': feature_cols,
                'test_accuracy': test_accuracy,
                'oob_accuracy': oob_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"\\nImproved model saved to {self.model_path}")
            
            return test_accuracy
            
        except Exception as e:
            print(f"Error training improved model: {e}")
            raise
    
    def _load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.feature_names = data['feature_names']
                self.test_accuracy = data.get('test_accuracy', 0)
                self.oob_accuracy = data.get('oob_accuracy', 0)
            print(f"Improved model loaded from {self.model_path}")
            print(f"Model test accuracy: {self.test_accuracy*100:.2f}%")
        except Exception as e:
            print(f"Error loading model: {e}. Retraining...")
            self._train_improved_model()
    
    def _create_features_for_prediction(self, n, p, k):
        """Create the same advanced features for prediction"""
        # Create DataFrame
        data = {'N': [n], 'P': [p], 'K': [k]}
        df = pd.DataFrame(data)
        
        # Apply same feature engineering
        df['N_P_ratio'] = df['N'] / (df['P'] + 0.1)
        df['N_K_ratio'] = df['N'] / (df['K'] + 0.1)
        df['P_K_ratio'] = df['P'] / (df['K'] + 0.1)
        
        df['P_N_ratio'] = df['P'] / (df['N'] + 0.1)
        df['K_N_ratio'] = df['K'] / (df['N'] + 0.1)
        df['K_P_ratio'] = df['K'] / (df['P'] + 0.1)
        
        df['NPK_total'] = df['N'] + df['P'] + df['K']
        df['NPK_avg'] = df['NPK_total'] / 3
        
        df['N_prop'] = df['N'] / df['NPK_total']
        df['P_prop'] = df['P'] / df['NPK_total']
        df['K_prop'] = df['K'] / df['NPK_total']
        
        df['N_sq'] = df['N'] ** 2
        df['P_sq'] = df['P'] ** 2
        df['K_sq'] = df['K'] ** 2
        
        df['N_sqrt'] = np.sqrt(df['N'])
        df['P_sqrt'] = np.sqrt(df['P'])
        df['K_sqrt'] = np.sqrt(df['K'])
        
        df['NP_product'] = df['N'] * df['P']
        df['NK_product'] = df['N'] * df['K']
        df['PK_product'] = df['P'] * df['K']
        
        df['max_nutrient'] = df[['N', 'P', 'K']].max(axis=1)
        df['min_nutrient'] = df[['N', 'P', 'K']].min(axis=1)
        df['nutrient_range'] = df['max_nutrient'] - df['min_nutrient']
        df['nutrient_std'] = df[['N', 'P', 'K']].std(axis=1)
        
        df['N_log'] = np.log1p(df['N'])
        df['P_log'] = np.log1p(df['P'])
        df['K_log'] = np.log1p(df['K'])
        
        return df
    
    def analyze_soil(self, n, p, k):
        if self.model is None or self.scaler is None:
            return {'error': 'Improved model is not loaded properly.'}
        
        try:
            # Create features for prediction
            feature_df = self._create_features_for_prediction(n, p, k)
            
            # Scale features
            scaled_data = self.scaler.transform(feature_df)
            
            # Make prediction
            crop_prediction = self.model.predict(scaled_data)[0]
            probabilities = self.model.predict_proba(scaled_data)[0]
            
            # Get top recommendations
            top_indices = np.argsort(probabilities)[-5:][::-1]
            recommended_crops = []
            
            for i in top_indices:
                crop_name = self.model.classes_[i]
                confidence = probabilities[i] * 100
                if confidence > 1.0:
                    recommended_crops.append({
                        "crop": crop_name, 
                        "confidence": f"{confidence:.1f}%"
                    })
            
            soil_quality = self._determine_soil_quality(n, p, k)
            
            return {
                'soil_quality': soil_quality,
                'recommended_crops': recommended_crops[:5],
                'top_crop': crop_prediction,
                'model_type': 'Improved Random Forest with Advanced Features',
                'model_accuracy': f"{getattr(self, 'test_accuracy', 0.64)*100:.2f}%"
            }
            
        except Exception as e:
            print(f"Error during improved analysis: {e}")
            return {'error': 'Improved analysis failed.'}
    
    def _determine_soil_quality(self, n, p, k):
        # Advanced soil quality assessment
        n_score = max(0, min(n / 150.0, 1))
        p_score = max(0, min(p / 75.0, 1))
        k_score = max(0, min(k / 200.0, 1))
        
        # Nutrient balance assessment
        total = n + p + k
        if total > 0:
            n_prop = n / total
            p_prop = p / total
            k_prop = k / total
            
            # Penalize extreme imbalances
            balance_score = 1 - (abs(n_prop - 0.33) + abs(p_prop - 0.33) + abs(k_prop - 0.33))
            balance_score = max(0, balance_score)
        else:
            balance_score = 0
        
        # Overall score with balance consideration
        level_score = (n_score + p_score + k_score) / 3
        final_score = level_score * 0.7 + balance_score * 0.3
        
        if final_score >= 0.8: return "Excellent"
        elif final_score >= 0.65: return "Good"
        elif final_score >= 0.45: return "Average"
        else: return "Poor"

if __name__ == "__main__":
    print("Testing Improved NPK Analyzer...")
    analyzer = ImprovedNPKAnalyzer()
    
    # Test with multiple sample values
    test_cases = [
        (90, 42, 43),   # Rice-like
        (150, 60, 70),  # Wheat-like
        (180, 100, 220) # Tomato-like
    ]
    
    for i, (n, p, k) in enumerate(test_cases, 1):
        result = analyzer.analyze_soil(n, p, k)
        print(f"\\nTest case {i} (N={n}, P={p}, K={k}): {result}")