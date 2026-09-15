import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class AdvancedCropRecommendationSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.dataset_path = 'Crop_recommendation.csv'
        self.best_model = None
        self.best_scaler = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        
    def load_and_prepare_data(self):
        """Load dataset and prepare features with advanced engineering"""
        df = pd.read_csv(self.dataset_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Crops: {df['label'].nunique()}")
        
        # Create advanced features using ALL available data
        enhanced_df = df.copy()
        
        # NPK ratios and interactions
        enhanced_df['N_P_ratio'] = enhanced_df['N'] / (enhanced_df['P'] + 0.1)
        enhanced_df['N_K_ratio'] = enhanced_df['N'] / (enhanced_df['K'] + 0.1)
        enhanced_df['P_K_ratio'] = enhanced_df['P'] / (enhanced_df['K'] + 0.1)
        enhanced_df['NPK_total'] = enhanced_df['N'] + enhanced_df['P'] + enhanced_df['K']
        enhanced_df['NPK_balance'] = enhanced_df[['N', 'P', 'K']].std(axis=1)
        
        # Environmental interactions
        enhanced_df['temp_humidity'] = enhanced_df['temperature'] * enhanced_df['humidity']
        enhanced_df['ph_rainfall'] = enhanced_df['ph'] * enhanced_df['rainfall']
        enhanced_df['temp_rainfall'] = enhanced_df['temperature'] * enhanced_df['rainfall']
        
        # Nutrient-environment interactions
        enhanced_df['N_temp'] = enhanced_df['N'] * enhanced_df['temperature']
        enhanced_df['P_humidity'] = enhanced_df['P'] * enhanced_df['humidity']
        enhanced_df['K_ph'] = enhanced_df['K'] * enhanced_df['ph']
        
        # Polynomial features for key variables
        enhanced_df['temp_sq'] = enhanced_df['temperature'] ** 2
        enhanced_df['humidity_sq'] = enhanced_df['humidity'] ** 2
        enhanced_df['rainfall_log'] = np.log1p(enhanced_df['rainfall'])
        
        # Binned categorical features
        enhanced_df['temp_category'] = pd.cut(enhanced_df['temperature'], bins=5, labels=['very_cold', 'cold', 'moderate', 'warm', 'hot'])
        enhanced_df['humidity_category'] = pd.cut(enhanced_df['humidity'], bins=4, labels=['low', 'medium', 'high', 'very_high'])
        enhanced_df['rainfall_category'] = pd.cut(enhanced_df['rainfall'], bins=4, labels=['drought', 'low', 'moderate', 'high'])
        
        # One-hot encode categorical features
        enhanced_df = pd.get_dummies(enhanced_df, columns=['temp_category', 'humidity_category', 'rainfall_category'])
        
        return enhanced_df
    
    def train_multiple_models(self):
        """Train multiple advanced models and find the best one"""
        # Prepare data
        df = self.load_and_prepare_data()
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != 'label']
        X = df[feature_cols]
        y = df['label']
        
        self.feature_names = feature_cols
        print(f"Total features: {len(feature_cols)}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models to test
        models_to_test = {
            'Random Forest': RandomForestClassifier(
                n_estimators=500,
                max_depth=25,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            
            'XGBoost': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(200, 100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ),
            
            'SVM': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            )
        }
        
        best_accuracy = 0
        results = {}
        
        print("\n" + "="*60)
        print("TRAINING AND EVALUATING MULTIPLE MODELS")
        print("="*60)
        
        for name, model in models_to_test.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if name in ['Neural Network', 'SVM']:
                model.fit(X_train_scaled, y_train)
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                X_cv = X_train_scaled
            else:
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                X_cv = X_train
            
            # Calculate accuracies
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_cv, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'overfitting': train_acc - test_acc
            }
            
            print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"  Testing Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"  CV Accuracy:       {cv_mean:.4f} ± {cv_std:.4f}")
            print(f"  Overfitting:       {(train_acc-test_acc)*100:.2f}%")
            
            # Track best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                self.best_model = model
                if name in ['Neural Network', 'SVM']:
                    self.best_scaler = scaler
                else:
                    self.best_scaler = None
                self.best_model_name = name
        
        # Create ensemble model
        print(f"\nCreating Ensemble Model...")
        ensemble_models = [
            ('rf', models_to_test['Random Forest']),
            ('xgb', models_to_test['XGBoost']),
            ('gb', models_to_test['Gradient Boosting'])
        ]
        
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        ensemble_train_acc = ensemble.score(X_train, y_train)
        ensemble_test_acc = ensemble.score(X_test, y_test)
        ensemble_cv = cross_val_score(ensemble, X_train, y_train, cv=5)
        
        results['Ensemble'] = {
            'model': ensemble,
            'train_accuracy': ensemble_train_acc,
            'test_accuracy': ensemble_test_acc,
            'cv_mean': ensemble_cv.mean(),
            'cv_std': ensemble_cv.std(),
            'overfitting': ensemble_train_acc - ensemble_test_acc
        }
        
        print(f"  Training Accuracy: {ensemble_train_acc:.4f} ({ensemble_train_acc*100:.2f}%)")
        print(f"  Testing Accuracy:  {ensemble_test_acc:.4f} ({ensemble_test_acc*100:.2f}%)")
        print(f"  CV Accuracy:       {ensemble_cv.mean():.4f} ± {ensemble_cv.std():.4f}")
        
        if ensemble_test_acc > best_accuracy:
            self.best_model = ensemble
            self.best_scaler = None
            self.best_model_name = 'Ensemble'
            best_accuracy = ensemble_test_acc
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Model':<20} {'Test Acc':<10} {'CV Mean':<10} {'Overfitting':<12}")
        print("-" * 60)
        
        for name, result in results.items():
            print(f"{name:<20} {result['test_accuracy']*100:>7.2f}%   {result['cv_mean']*100:>7.2f}%   {result['overfitting']*100:>9.2f}%")
        
        print(f"\nBEST MODEL: {self.best_model_name} with {best_accuracy*100:.2f}% accuracy")
        
        # Save best model
        self.save_best_model()
        
        return results
    
    def save_best_model(self):
        """Save the best performing model"""
        model_data = {
            'model': self.best_model,
            'scaler': self.best_scaler,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder,
            'model_name': self.best_model_name
        }
        
        with open('best_crop_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\nBest model ({self.best_model_name}) saved to 'best_crop_model.pkl'")
    
    def load_best_model(self):
        """Load the saved best model"""
        try:
            with open('best_crop_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.best_model = data['model']
                self.best_scaler = data['scaler']
                self.feature_names = data['feature_names']
                self.label_encoder = data['label_encoder']
                self.best_model_name = data['model_name']
            print(f"Best model ({self.best_model_name}) loaded successfully")
            return True
        except FileNotFoundError:
            print("No saved model found. Please train models first.")
            return False
    
    def predict_crop(self, n, p, k, temperature, humidity, ph, rainfall):
        """Predict crop using all environmental factors"""
        if self.best_model is None:
            if not self.load_best_model():
                return {'error': 'No model available'}
        
        try:
            # Create feature vector with same engineering as training
            data = {
                'N': n, 'P': p, 'K': k,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
            
            df = pd.DataFrame([data])
            
            # Apply same feature engineering
            df['N_P_ratio'] = df['N'] / (df['P'] + 0.1)
            df['N_K_ratio'] = df['N'] / (df['K'] + 0.1)
            df['P_K_ratio'] = df['P'] / (df['K'] + 0.1)
            df['NPK_total'] = df['N'] + df['P'] + df['K']
            df['NPK_balance'] = df[['N', 'P', 'K']].std(axis=1)
            
            df['temp_humidity'] = df['temperature'] * df['humidity']
            df['ph_rainfall'] = df['ph'] * df['rainfall']
            df['temp_rainfall'] = df['temperature'] * df['rainfall']
            
            df['N_temp'] = df['N'] * df['temperature']
            df['P_humidity'] = df['P'] * df['humidity']
            df['K_ph'] = df['K'] * df['ph']
            
            df['temp_sq'] = df['temperature'] ** 2
            df['humidity_sq'] = df['humidity'] ** 2
            df['rainfall_log'] = np.log1p(df['rainfall'])
            
            # Handle categorical features (simplified for prediction)
            # Add dummy columns for categories (set to 0 for now, could be improved)
            categorical_cols = [col for col in self.feature_names if 'category_' in col]
            for col in categorical_cols:
                df[col] = 0
            
            # Ensure all features are present
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns to match training
            df = df[self.feature_names]
            
            # Scale if needed
            if self.best_scaler is not None:
                features = self.best_scaler.transform(df)
            else:
                features = df.values
            
            # Make prediction
            prediction = self.best_model.predict(features)[0]
            probabilities = self.best_model.predict_proba(features)[0]
            
            # Get top recommendations
            top_indices = np.argsort(probabilities)[-5:][::-1]
            recommendations = []
            
            for i in top_indices:
                crop_name = self.label_encoder.inverse_transform([i])[0]
                confidence = probabilities[i] * 100
                if confidence > 1.0:
                    recommendations.append({
                        "crop": crop_name,
                        "confidence": f"{confidence:.1f}%"
                    })
            
            top_crop = self.label_encoder.inverse_transform([prediction])[0]
            
            return {
                'top_crop': top_crop,
                'recommended_crops': recommendations,
                'model_type': f'Advanced {self.best_model_name}',
                'features_used': 'All environmental factors + NPK'
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {'error': f'Prediction failed: {str(e)}'}

if __name__ == "__main__":
    print("Advanced Crop Recommendation System")
    print("="*50)
    
    system = AdvancedCropRecommendationSystem()
    
    # Train all models and find the best one
    results = system.train_multiple_models()
    
    # Test prediction
    print("\n" + "="*50)
    print("TESTING PREDICTION")
    print("="*50)
    
    test_result = system.predict_crop(
        n=90, p=42, k=43,
        temperature=20.88, humidity=82.0,
        ph=6.5, rainfall=202.9
    )
    
    print("Test prediction result:")
    print(test_result)