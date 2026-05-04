from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os

TRAINING_DATA_PATH = Path(__file__).parent / "Datasets" / "part-00000-0af89d10-df53-44fd-b124-a8a496fd5023-c000(2).csv"
EXCLUDE_COLUMNS = {"dest_ip", "src_ip", "uid", "community_id", "datetime", "ts"}
LOGISTIC_MODEL_PATH = Path(__file__).parent / "Models" / "logistic_regression_combined_output.joblib"
DECISION_TREE_MODEL_PATH = Path(__file__).parent / "Models" / "decision_tree_combined_output.joblib"
RANDOM_FOREST_MODEL_PATH = Path(__file__).parent / "Models" / "random_forest_combined_output.joblib"


def engineer_features(df):
    """Apply feature engineering to match the training pipeline"""
    df = df.copy()
    
    # Total packets and bytes
    df['total_pkts'] = df['orig_pkts'] + df['resp_pkts']
    df['total_bytes'] = df['orig_bytes'] + df['resp_bytes']
    
    # Packet and byte ratios (handling division by zero)
    df['pkt_ratio'] = df['resp_pkts'] / df['total_pkts'].replace(0, np.nan)
    df['byte_ratio'] = df['resp_bytes'] / df['total_bytes'].replace(0, np.nan)
    df[['pkt_ratio', 'byte_ratio']] = df[['pkt_ratio', 'byte_ratio']].fillna(0)
    
    # Packet and byte rates (handling division by zero and infinite values)
    df['pkt_rate'] = df['total_pkts'] / df['duration'].replace(0, np.nan)
    df['byte_rate'] = df['total_bytes'] / df['duration'].replace(0, np.nan)
    df[['pkt_rate', 'byte_rate']] = df[['pkt_rate', 'byte_rate']].replace([np.inf, -np.inf], 0).fillna(0)
    
    return df


def keep_model_columns(df, model):
    """Keep only columns expected by the trained model and drop unseen uploaded columns."""
    if not hasattr(model, 'feature_names_in_'):
        return df
    required_columns = list(model.feature_names_in_)
    return df.loc[:, df.columns.intersection(required_columns)]


def preprocess_for_logistic(df):
    """
    Preprocess data for Logistic Regression model.
    Matches the preprocessing from SOCLogsLogisticRegression.ipynb
    """
    df = df.copy()

    # Remove target column if present
    if 'mitre_attack_tactics' in df.columns:
        df = df.drop(columns=['mitre_attack_tactics'])

    # Drop excluded columns
    df = df.drop(columns=[col for col in EXCLUDE_COLUMNS if col in df.columns], errors='ignore')

    # Convert numeric-looking strings to numbers
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() > len(df) * 0.75:
                df[col] = converted

    # Apply feature engineering
    df = engineer_features(df)

    # Handle missing values in categorical features
    categorical_features = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    if categorical_features:
        df[categorical_features] = df[categorical_features].fillna("missing")

    return df


def preprocess_for_decision_tree(df):
    """
    Preprocess data for Decision Tree model.
    Matches the preprocessing from SOCDecisionTree.ipynb exactly
    """
    df = df.copy()

    # Remove target column if present
    if 'mitre_attack_tactics' in df.columns:
        df = df.drop(columns=['mitre_attack_tactics'])

    # Drop excluded columns
    df = df.drop(columns=[col for col in EXCLUDE_COLUMNS if col in df.columns], errors='ignore')

    # Define the exact features used in the Decision Tree notebook
    numerical_features = [
        'resp_pkts', 'orig_ip_bytes', 'missed_bytes', 'duration',
        'orig_pkts', 'resp_ip_bytes', 'dest_port', 'orig_bytes', 'resp_bytes', 'src_port'
    ]
    categorical_features = [
        'service', 'protocol', 'conn_state', 'local_resp', 'local_orig'
    ]

    # Keep only the features used in training
    features = numerical_features + categorical_features
    df = df[features].copy()

    # Apply feature engineering (exactly as in the notebook)
    df['total_pkts'] = df['orig_pkts'] + df['resp_pkts']
    df['total_bytes'] = df['orig_bytes'] + df['resp_bytes']
    df['pkt_ratio'] = df['resp_pkts'] / df['total_pkts'].replace(0, np.nan)
    df['byte_ratio'] = df['resp_bytes'] / df['total_bytes'].replace(0, np.nan)
    df[['pkt_ratio', 'byte_ratio']] = df[['pkt_ratio', 'byte_ratio']].fillna(0)
    df['pkt_rate'] = df['total_pkts'] / df['duration'].replace(0, np.nan)
    df['byte_rate'] = df['total_bytes'] / df['duration'].replace(0, np.nan)
    df[['pkt_rate', 'byte_rate']] = df[['pkt_rate', 'byte_rate']].replace([np.inf, -np.inf], 0).fillna(0)

    # Handle missing values in categorical features
    for cat_col in categorical_features:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].fillna('missing')

    # Create one-hot encoded features manually to match training exactly
    # Based on the combined training data categories

    # Service encoding (reference: dhcp, so we create dns and ntp)
    df['service_dns'] = (df['service'] == 'dns').astype(int)
    df['service_ntp'] = (df['service'] == 'ntp').astype(int)

    # Protocol encoding (reference: icmp, so we don't create icmp but create others)
    # The model expects protocol_udp but not protocol_tcp, so protocol_udp=1 when protocol='udp'
    df['protocol_udp'] = (df['protocol'] == 'udp').astype(int)

    # Conn_state encoding (reference: OTH, so we create SF and SHR)
    df['conn_state_SF'] = (df['conn_state'] == 'SF').astype(int)
    df['conn_state_SHR'] = (df['conn_state'] == 'SHR').astype(int)

    # Local_resp and local_orig (boolean to int)
    df['local_resp_True'] = df['local_resp'].astype(int)
    df['local_orig_True'] = df['local_orig'].astype(int)

    # Drop the original categorical columns
    df = df.drop(columns=categorical_features)

    # Ensure all expected features are present (add missing ones as 0)
    expected_features = [
        'resp_pkts', 'orig_ip_bytes', 'missed_bytes', 'duration', 'orig_pkts',
        'resp_ip_bytes', 'dest_port', 'orig_bytes', 'resp_bytes', 'src_port',
        'total_pkts', 'total_bytes', 'pkt_ratio', 'byte_ratio', 'pkt_rate', 'byte_rate',
        'service_dns', 'service_ntp', 'protocol_udp', 'conn_state_SF', 'conn_state_SHR',
        'local_resp_True', 'local_orig_True'
    ]

    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0

    # Reorder columns to match expected order
    df = df[expected_features]

    return df


def preprocess_for_random_forest(df):
    """
    Preprocess data for Random Forest model.
    Matches the preprocessing from SOCTransformers.ipynb
    """
    df = df.copy()

    # Remove target column if present
    if 'mitre_attack_tactics' in df.columns:
        df = df.drop(columns=['mitre_attack_tactics'])

    # Apply feature engineering first
    df = engineer_features(df)

    # Drop excluded columns (done after feature engineering in the notebook)
    df = df.drop(columns=[col for col in EXCLUDE_COLUMNS if col in df.columns], errors='ignore')

    # Identify feature types (like in the notebook)
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include='object').columns.tolist()
    boolean_features = df.select_dtypes(include='bool').columns.tolist()
    categorical_features.extend(boolean_features)

    # Refine categorical features: numerical columns with few unique values
    for col in numerical_features.copy():
        if df[col].nunique() < 10 and df[col].dtype == 'int64':
            categorical_features.append(col)
            numerical_features.remove(col)

    # Handle missing values
    if categorical_features:
        df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])

    return df


app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained models
logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
decision_tree_model = joblib.load(DECISION_TREE_MODEL_PATH)
random_forest_model = joblib.load(RANDOM_FOREST_MODEL_PATH)

# Keep a default model alias for existing model-info behavior
model = logistic_model


@app.route('/')
def index():
    """Main page for uploading log files"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analyze for reconnaissance attacks"""
    if 'logfile' not in request.files:
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['logfile']
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    
    if not file.filename.endswith('.csv'):
        return render_template('index.html', error='Please upload a CSV file')
    
    try:
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Load the CSV
        df = pd.read_csv(filepath)
        df_examples = df.copy()
        
        # Remove target column if present
        if 'mitre_attack_tactics' in df.columns:
            df = df.drop('mitre_attack_tactics', axis=1)
        
        # Preprocess for each model
        logistic_df = preprocess_for_logistic(df)
        logistic_df = keep_model_columns(logistic_df, logistic_model)

        decision_df = preprocess_for_decision_tree(df)
        decision_df = keep_model_columns(decision_df, decision_tree_model)

        random_forest_df = preprocess_for_random_forest(df)
        random_forest_df = keep_model_columns(random_forest_df, random_forest_model)
        
        # Make predictions
        logistic_preds = logistic_model.predict(logistic_df)
        decision_preds = decision_tree_model.predict(decision_df)
        random_forest_preds = random_forest_model.predict(random_forest_df)
        
        # Count reconnaissance attacks for each model
        # Logistic returns text labels
        logistic_recon_count = sum(1 for pred in logistic_preds if 'Reconnaissance' in str(pred))
        # Decision tree returns numeric labels (1 = Reconnaissance, 0 = none)
        decision_recon_count = sum(1 for pred in decision_preds if pred == 1)
        # Random forest returns text labels
        random_forest_recon_count = sum(1 for pred in random_forest_preds if 'Reconnaissance' in str(pred))
        
        total_entries = len(logistic_preds)
        logistic_percentage = round((logistic_recon_count / total_entries) * 100, 2) if total_entries > 0 else 0
        decision_percentage = round((decision_recon_count / total_entries) * 100, 2) if total_entries > 0 else 0
        random_forest_percentage = round((random_forest_recon_count / total_entries) * 100, 2) if total_entries > 0 else 0
        
        # Get up to 5 examples of reconnaissance attacks for each model
        logistic_indices = [i for i, pred in enumerate(logistic_preds) if 'Reconnaissance' in str(pred)]
        decision_indices = [i for i, pred in enumerate(decision_preds) if pred == 1]
        random_forest_indices = [i for i, pred in enumerate(random_forest_preds) if 'Reconnaissance' in str(pred)]
        logistic_examples = df_examples.iloc[logistic_indices[:5]].to_dict('records') if logistic_indices else []
        decision_examples = df_examples.iloc[decision_indices[:5]].to_dict('records') if decision_indices else []
        random_forest_examples = df_examples.iloc[random_forest_indices[:5]].to_dict('records') if random_forest_indices else []
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return render_template('index.html', 
                             results=True,
                             total_entries=total_entries,
                             logistic_reconnaissance_count=logistic_recon_count,
                             logistic_percentage=logistic_percentage,
                             decision_reconnaissance_count=decision_recon_count,
                             decision_percentage=decision_percentage,
                             random_forest_reconnaissance_count=random_forest_recon_count,
                             random_forest_percentage=random_forest_percentage,
                             logistic_examples=logistic_examples,
                             decision_examples=decision_examples,
                             random_forest_examples=random_forest_examples)
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return render_template('index.html', error=f'Error processing file: {str(e)}')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions using the SOC logistic model.
    Expects JSON input with features similar to the training data.
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Convert to DataFrame (assuming single prediction or batch)
        if isinstance(data, dict):
            # Single prediction
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            # Batch prediction
            df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid data format. Expected dict or list of dicts'}), 400

        # Apply preprocessing for each model
        logistic_df = preprocess_for_logistic(df)
        logistic_df = keep_model_columns(logistic_df, logistic_model)

        decision_df = preprocess_for_decision_tree(df)
        decision_df = keep_model_columns(decision_df, decision_tree_model)

        random_forest_df = preprocess_for_random_forest(df)
        random_forest_df = keep_model_columns(random_forest_df, random_forest_model)
        
        # Make predictions
        logistic_preds = logistic_model.predict(logistic_df)
        decision_preds = decision_tree_model.predict(decision_df)
        random_forest_preds = random_forest_model.predict(random_forest_df)

        # Get prediction probabilities for the logistic pipeline if supported
        try:
            logistic_probabilities = logistic_model.predict_proba(logistic_df)
            logistic_classes = logistic_model.classes_
        except:
            logistic_probabilities = None
            logistic_classes = None

        response = {
            'logistic_predictions': logistic_preds.tolist(),
            'decision_tree_predictions': decision_preds.tolist(),
            'random_forest_predictions': random_forest_preds.tolist(),
        }

        if logistic_probabilities is not None and logistic_classes is not None:
            response['logistic_probabilities'] = logistic_probabilities.tolist()
            response['logistic_classes'] = logistic_classes.tolist()

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

@app.route('/model-info', methods=['GET'])
def model_info():
    """Display comprehensive information about the loaded model"""
    try:
        print(type(model).__name__)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SOC Model Information</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px; }}
                .section {{ margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 5px; }}
                .feature-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px; }}
                .feature-item {{ padding: 10px; background: white; border-radius: 3px; border-left: 3px solid #27ae60; }}
                .coef-positive {{ color: #e74c3c; font-weight: bold; }}
                .coef-negative {{ color: #3498db; font-weight: bold; }}
                .metric {{ display: inline-block; background: #ecf0f1; padding: 5px 10px; margin: 2px; border-radius: 3px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; font-weight: bold; }}
                .highlight {{ background-color: #fff3cd; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 SOC Model Analysis Dashboard</h1>
                <p><strong>Model Type:</strong> {type(model).__name__}</p>
                <p><strong>Model Path:</strong> {LOGISTIC_MODEL_PATH}</p>
        """

        # Pipeline information
        if isinstance(model, Pipeline):
            html_content += """
                <h2>📋 Pipeline Structure</h2>
                <div class="section">
                    <ol>
            """

            for i, (name, step) in enumerate(model.named_steps.items()):
                html_content += f"<li><strong>{name}:</strong> {type(step).__name__}</li>"

            html_content += "</ol></div>"

            # Preprocessor details
            preprocessor = model.named_steps.get('preprocessor')
            if preprocessor and isinstance(preprocessor, ColumnTransformer):
                html_content += """
                    <h2>🔧 Preprocessor Details</h2>
                    <div class="section">
                """

                for name, transformer, columns in preprocessor.transformers_:
                    if transformer != 'drop':
                        html_content += f"""
                            <h3>Transformer: {name} ({type(transformer).__name__})</h3>
                            <p><strong>Columns:</strong> {', '.join(columns)}</p>
                        """

                        if hasattr(transformer, 'categories_') and transformer.categories_:
                            html_content += "<p><strong>Categories:</strong></p><ul>"
                            for cat_list in transformer.categories_[:3]:  # Show first 3 categories
                                cat_preview = ', '.join(str(cat) for cat in cat_list[:5])
                                if len(cat_list) > 5:
                                    cat_preview += "..."
                                html_content += f"<li>{cat_preview}</li>"
                            html_content += "</ul>"

                html_content += "</div>"

        # Get the final estimator
        final_estimator = model.named_steps[list(model.named_steps.keys())[-1]] if isinstance(model, Pipeline) else model

        if isinstance(final_estimator, LogisticRegression):
            html_content += f"""
                <h2>📊 Logistic Regression Details</h2>
                <div class="section">
                    <h3>Model Parameters</h3>
                    <div class="feature-list">
            """

            # Key parameters
            key_params = ['C', 'class_weight', 'solver', 'max_iter', 'random_state']
            for param in key_params:
                value = getattr(final_estimator, param, 'N/A')
                html_content += f'<div class="metric"><strong>{param}:</strong> {value}</div>'

            html_content += f"""
                    </div>

                    <h3>Model Attributes</h3>
                    <table>
                        <tr><th>Attribute</th><th>Value</th></tr>
                        <tr><td>Classes</td><td>{', '.join(final_estimator.classes_)}</td></tr>
                        <tr><td>Number of Features</td><td>{final_estimator.n_features_in_}</td></tr>
                        <tr><td>Number of Iterations</td><td>{final_estimator.n_iter_[0]}</td></tr>
                        <tr class="highlight"><td>Coefficients Shape</td><td>{final_estimator.coef_.shape}</td></tr>
                    </table>
                </div>
            """

            # Feature importance
            if hasattr(final_estimator, 'coef_'):
                coef = final_estimator.coef_[0]

                # Get feature names
                feature_names = []
                if isinstance(model, Pipeline) and preprocessor:
                    try:
                        feature_names = list(preprocessor.get_feature_names_out())
                    except:
                        feature_names = [f"feature_{i}" for i in range(len(coef))]

                if feature_names:
                    coef_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': coef,
                        'Abs_Coefficient': np.abs(coef)
                    }).sort_values('Abs_Coefficient', ascending=False)

                    html_content += """
                        <h2>🔝 Top 15 Most Important Features</h2>
                        <div class="section">
                            <table>
                                <tr><th>Rank</th><th>Feature</th><th>Coefficient</th><th>Direction</th></tr>
                    """

                    for idx, row in coef_df.head(15).iterrows():
                        direction = "↑ Increases Reconnaissance" if row['Coefficient'] > 0 else "↓ Decreases Reconnaissance"
                        coef_class = "coef-positive" if row['Coefficient'] > 0 else "coef-negative"
                        html_content += f"""
                            <tr>
                                <td>{idx+1}</td>
                                <td>{row['Feature'][:30]}{'...' if len(str(row['Feature'])) > 30 else ''}</td>
                                <td class="{coef_class}">{row['Coefficient']:.4f}</td>
                                <td>{direction}</td>
                            </tr>
                        """

                    html_content += "</table></div>"

        # Performance info
        html_content += """
            <h2>📈 Model Performance</h2>
            <div class="section">
                <p><strong>Note:</strong> For detailed performance metrics, run the training script.</p>
                <p><strong>Classification Task:</strong> The model distinguishes between SOC events:</p>
                <ul>
        """

        if hasattr(final_estimator, 'classes_'):
            for cls in final_estimator.classes_:
                html_content += f"<li><strong>{cls}</strong></li>"

        html_content += """
                </ul>
                <div style="background: #e8f4f8; padding: 15px; border-radius: 5px; margin-top: 20px;">
                    <h4>💡 Usage Tips:</h4>
                    <ul>
                        <li><strong style="color: #e74c3c;">Red coefficients:</strong> Features that increase 'Reconnaissance' probability</li>
                        <li><strong style="color: #3498db;">Blue coefficients:</strong> Features that decrease 'Reconnaissance' probability</li>
                        <li><strong>Higher absolute coefficient = More important feature</strong></li>
                    </ul>
                </div>
            </div>
        </div>
        </body>
        </html>
        """

        return html_content

    except Exception as e:
        return f"<h1>Error loading model information</h1><p>{str(e)}</p>", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)