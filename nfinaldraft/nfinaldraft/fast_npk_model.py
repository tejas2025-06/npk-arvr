import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class FastNPKModel:
    def __init__(self):
        self.dataset_path = 'Crop_recommendation.csv'
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.best_accuracy = 0
        
    def create_smart_features(self, df):
        """Create the most effective features based on domain knowledge"""
        # Start with NPK only
        enhanced_df = df[['N', 'P', 'K', 'label']].copy()
        
        # Most important ratios (proven effective)
        enhanced_df['N_P_ratio'] = enhanced_df['N'] / (enhanced_df['P'] + 0.01)
        enhanced_df['N_K_ratio'] = enhanced_df['N'] / (enhanced_df['K'] + 0.01)
        enhanced_df['P_K_ratio'] = enhanced_df['P'] / (enhanced_df['K'] + 0.01)
        enhanced_df['K_P_ratio'] = enhanced_df['K'] / (enhanced_df['P'] + 0.01)
        enhanced_df['K_N_ratio'] = enhanced_df['K'] / (enhanced_df['N'] + 0.01)
        enhanced_df['P_N_ratio'] = enhanced_df['P'] / (enhanced_df['N'] + 0.01)
        
        # Totals and statistics
        enhanced_df['NPK_total'] = enhanced_df['N'] + enhanced_df['P'] + enhanced_df['K']
        enhanced_df['NPK_avg'] = enhanced_df['NPK_total'] / 3
        enhanced_df['NPK_std'] = enhanced_df[['N', 'P', 'K']].std(axis=1)
        enhanced_df['NPK_max'] = enhanced_df[['N', 'P', 'K']].max(axis=1)
        enhanced_df['NPK_min'] = enhanced_df[['N', 'P', 'K']].min(axis=1)
        
        # Proportions (very important for crop selection)
        enhanced_df['N_prop'] = enhanced_df['N'] / enhanced_df['NPK_total']
        enhanced_df['P_prop'] = enhanced_df['P'] / enhanced_df['NPK_total']
        enhanced_df['K_prop'] = enhanced_df['K'] / enhanced_df['NPK_total']
        
        # Power transformations (most effective)
        enhanced_df['N_sq'] = enhanced_df['N'] ** 2
        enhanced_df['P_sq'] = enhanced_df['P'] ** 2
        enhanced_df['K_sq'] = enhanced_df['K'] ** 2
        enhanced_df['N_sqrt'] = np.sqrt(enhanced_df['N'])
        enhanced_df['P_sqrt'] = np.sqrt(enhanced_df['P'])
        enhanced_df['K_sqrt'] = np.sqrt(enhanced_df['K'])
        
        # Log transformations
        enhanced_df['N_log'] = np.log1p(enhanced_df['N'])
        enhanced_df['P_log'] = np.log1p(enhanced_df['P'])
        enhanced_df['K_log'] = np.log1p(enhanced_df['K'])
        
        # Key interactions
        enhanced_df['NP_product'] = enhanced_df['N'] * enhanced_df['P']
        enhanced_df['NK_product'] = enhanced_df['N'] * enhanced_df['K']
        enhanced_df['PK_product'] = enhanced_df['P'] * enhanced_df['K']
        
        # Balance indicators
        enhanced_df['balance_score'] = 1 - (enhanced_df['NPK_std'] / (enhanced_df['NPK_avg'] + 0.01))
        enhanced_df['N_dominance'] = enhanced_df['N'] / enhanced_df['NPK_max']
        enhanced_df['P_dominance'] = enhanced_df['P'] / enhanced_df['NPK_max']
        enhanced_df['K_dominance'] = enhanced_df['K'] / enhanced_df['NPK_max']
        
        # Advanced ratios
        enhanced_df['N_over_PK'] = enhanced_df['N'] / (enhanced_df['P'] + enhanced_df['K'] + 0.01)
        enhanced_df['P_over_NK'] = enhanced_df['P'] / (enhanced_df['N'] + enhanced_df['K'] + 0.01)
        enhanced_df['K_over_NP'] = enhanced_df['K'] / (enhanced_df['N'] + enhanced_df['P'] + 0.01)
        
        # Geometric mean
        enhanced_df['NPK_geom_mean'] = (enhanced_df['N'] * enhanced_df['P'] * enhanced_df['K']) ** (1/3)
        
        # Replace inf values
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        
        return enhanced_df
    
    def train_fast_model(self):
        """Train models quickly with pre-optimized parameters"""
        # Load and prepare data
        df = pd.read_csv(self.dataset_path)
        print(f"Dataset shape: {df.shape}")
        
        # Create features
        enhanced_df = self.create_smart_features(df)
        print(f"Enhanced shape: {enhanced_df.shape}")
        
        # Prepare features and target
        feature_cols = [col for col in enhanced_df.columns if col != 'label']
        X = enhanced_df[feature_cols]
        y = enhanced_df['label']
        
        # Handle NaN values
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=feature_cols)
        
        self.feature_names = feature_cols
        print(f"Features: {len(feature_cols)}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data (larger test set for better accuracy estimation)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
        )
        
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        models_results = {}
        
        print("\n" + "="*50)
        print("TRAINING FAST OPTIMIZED MODELS")
        print("="*50)
        
        # 1. Optimized Random Forest (pre-tuned parameters)
        print("1. Training Optimized Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=1500,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        rf_model.fit(X_train, y_train)
        rf_test_acc = rf_model.score(X_test, y_test)
        rf_oob_acc = rf_model.oob_score_
        rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
        
        models_results['RandomForest'] = {
            'model': rf_model,
            'test_accuracy': rf_test_acc,
            'oob_accuracy': rf_oob_acc,
            'cv_mean': rf_cv_scores.mean(),
            'cv_std': rf_cv_scores.std()
        }
        
        print(f"  Test Accuracy: {rf_test_acc:.4f} ({rf_test_acc*100:.2f}%)")
        print(f"  OOB Accuracy:  {rf_oob_acc:.4f} ({rf_oob_acc*100:.2f}%)")
        print(f"  CV Accuracy:   {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
        
        # 2. Optimized Extra Trees
        print("\n2. Training Optimized Extra Trees...")
        et_model = ExtraTreesClassifier(
            n_estimators=1200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        et_model.fit(X_train, y_train)
        et_test_acc = et_model.score(X_test, y_test)
        et_oob_acc = et_model.oob_score_
        et_cv_scores = cross_val_score(et_model, X_train, y_train, cv=5, scoring='accuracy')
        
        models_results['ExtraTrees'] = {
            'model': et_model,
            'test_accuracy': et_test_acc,
            'oob_accuracy': et_oob_acc,
            'cv_mean': et_cv_scores.mean(),
            'cv_std': et_cv_scores.std()
        }
        
        print(f"  Test Accuracy: {et_test_acc:.4f} ({et_test_acc*100:.2f}%)")
        print(f"  OOB Accuracy:  {et_oob_acc:.4f} ({et_oob_acc*100:.2f}%)")
        print(f"  CV Accuracy:   {et_cv_scores.mean():.4f} ± {et_cv_scores.std():.4f}")
        
        # 3. Optimized XGBoost
        print("\n3. Training Optimized XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(X_train, y_train)
        xgb_test_acc = xgb_model.score(X_test, y_test)
        xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy')
        
        models_results['XGBoost'] = {
            'model': xgb_model,
            'test_accuracy': xgb_test_acc,
            'cv_mean': xgb_cv_scores.mean(),
            'cv_std': xgb_cv_scores.std()
        }
        
        print(f"  Test Accuracy: {xgb_test_acc:.4f} ({xgb_test_acc*100:.2f}%)")
        print(f"  CV Accuracy:   {xgb_cv_scores.mean():.4f} ± {xgb_cv_scores.std():.4f}")
        
        # 4. Create Ensemble
        print("\n4. Creating Ensemble...")
        
        # Convert back to string labels for ensemble
        y_train_str = self.label_encoder.inverse_transform(y_train)
        y_test_str = self.label_encoder.inverse_transform(y_test)
        
        ensemble_models = [
            ('rf', rf_model),
            ('et', et_model)
        ]
        
        ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
        ensemble.fit(X_train, y_train_str)
        
        ensemble_test_acc = ensemble.score(X_test, y_test_str)
        ensemble_cv_scores = cross_val_score(ensemble, X_train, y_train_str, cv=5, scoring='accuracy')
        
        models_results['Ensemble'] = {
            'model': ensemble,
            'test_accuracy': ensemble_test_acc,
            'cv_mean': ensemble_cv_scores.mean(),
            'cv_std': ensemble_cv_scores.std()
        }
        
        print(f"  Test Accuracy: {ensemble_test_acc:.4f} ({ensemble_test_acc*100:.2f}%)")
        print(f"  CV Accuracy:   {ensemble_cv_scores.mean():.4f} ± {ensemble_cv_scores.std():.4f}")
        
        # Find best model
        best_model_name = max(models_results.keys(), key=lambda k: models_results[k]['test_accuracy'])
        self.best_accuracy = models_results[best_model_name]['test_accuracy']
        self.model = models_results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        # Print results
        print(f"\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        print(f"{'Model':<15} {'Test Acc':<10} {'CV Mean':<10}")
        print("-" * 40)
        
        for name, result in models_results.items():
            print(f"{name:<15} {result['test_accuracy']*100:>7.2f}%   {result['cv_mean']*100:>7.2f}%")
        
        print(f"\nBEST: {best_model_name} - {self.best_accuracy*100:.2f}%")
        
        if self.best_accuracy >= 0.80:
            print("\n🎉 SUCCESS: 80%+ accuracy achieved!")
        elif self.best_accuracy >= 0.75:
            print(f"\n📈 CLOSE: {self.best_accuracy*100:.2f}% (need {80-self.best_accuracy*100:.1f}% more)")
        else:
            print(f"\n📊 Current: {self.best_accuracy*100:.2f}% (target: 80%+)")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTOP 10 FEATURES:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<15}: {row['importance']:.4f}")
        
        # Save model
        self.save_model()
        
        return self.best_accuracy
    
    def save_model(self):
        """Save the best model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder,
            'accuracy': self.best_accuracy,
            'model_name': self.best_model_name
        }
        
        with open('fast_npk_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\nModel saved to 'fast_npk_model.pkl'")
    
    def load_model(self):
        """Load saved model"""
        try:
            with open('fast_npk_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data.get('scaler')
                self.imputer = data['imputer']
                self.feature_names = data['feature_names']
                self.label_encoder = data['label_encoder']
                self.best_accuracy = data['accuracy']
                self.best_model_name = data['model_name']
            return True
        except FileNotFoundError:
            return False
    
    def predict(self, n, p, k):
        """Make prediction using NPK values only"""
        if self.model is None:
            if not self.load_model():
                return {'error': 'No model available'}
        
        try:
            # Create feature vector
            data = pd.DataFrame({'N': [n], 'P': [p], 'K': [k], 'label': ['dummy']})
            feature_df = self.create_smart_features(data)
            
            # Remove label and get features in correct order
            feature_df = feature_df.drop('label', axis=1)
            feature_df = feature_df[self.feature_names]
            
            # Handle NaN values
            feature_df = pd.DataFrame(
                self.imputer.transform(feature_df),
                columns=self.feature_names
            )
            
            # Make prediction
            if 'XGBoost' in self.best_model_name:
                prediction_encoded = self.model.predict(feature_df)[0]
                probabilities = self.model.predict_proba(feature_df)[0]
                prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
                class_names = self.label_encoder.inverse_transform(range(len(probabilities)))
            else:
                prediction = self.model.predict(feature_df)[0]
                probabilities = self.model.predict_proba(feature_df)[0]
                class_names = self.model.classes_
            
            # Get top recommendations
            top_indices = np.argsort(probabilities)[-5:][::-1]
            recommendations = []
            
            for i in top_indices:
                crop_name = class_names[i]
                confidence = probabilities[i] * 100
                if confidence > 1.0:
                    recommendations.append({
                        "crop": crop_name,
                        "confidence": f"{confidence:.1f}%"
                    })
            
            return {
                'top_crop': prediction,
                'recommended_crops': recommendations,
                'model_accuracy': f"{self.best_accuracy*100:.2f}%",
                'model_type': f'Fast NPK Model ({self.best_model_name})',
                'features_used': f'{len(self.feature_names)} NPK features'
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}

if __name__ == "__main__":
    print("Fast NPK Model - Quick Training for 80%+ Accuracy")
    print("="*50)
    
    model = FastNPKModel()
    accuracy = model.train_fast_model()
    
    # Test prediction
    print(f"\n" + "="*50)
    print("TESTING PREDICTION")
    print("="*50)
    
    test_result = model.predict(90, 42, 43)
    print("Test (N=90, P=42, K=43):")
    for key, value in test_result.items():
        print(f"  {key}: {value}")