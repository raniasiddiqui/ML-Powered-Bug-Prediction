import streamlit as st
import pandas as pd
import requests
import numpy as np
from requests.auth import HTTPBasicAuth
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from groq import Groq
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import json as pyjson
import re
import pandas as pd

# ================================
# PATHS
# ================================
MODEL_DIR = Path("models")
OUT_DIR = Path("results")
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ================================
# CONFIG & INITIALIZATION
# ================================
@st.cache_resource
# ================================
# CONFIG & INITIALIZATION
# ================================
@st.cache_resource
def load_config():
    required_keys = ["ORG", "PAT", "groq_api_key", "groq_default_model"]
    missing = [k for k in required_keys if k not in st.secrets]

    if missing:
        st.error(f"❌ Missing keys in secrets.toml: {', '.join(missing)}")
        st.stop()

    return {
        "ORG": st.secrets["ORG"],
        "PAT": st.secrets["PAT"],
        "groq_api_key": st.secrets["groq_api_key"],
        "groq_default_model": st.secrets["groq_default_model"],
    }

config = load_config()
ORG = config["ORG"]
PAT = config["PAT"]
auth = HTTPBasicAuth("", PAT)

groq_api_key = config.get("groq_api_key", "").strip()
groq_model = config.get("groq_default_model", "llama3-70b-8192")

if not groq_api_key:
    st.error("⚠️ groq_api_key missing in secrets.toml")
    st.stop()

@st.cache_resource
def init_groq_client():
    return Groq(api_key=groq_api_key)

client = init_groq_client()

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

# ================================
# FETCH & PREPROCESS
# ================================

@st.cache_data(ttl=3600)
def fetch_projects():
    url = f"https://dev.azure.com/{ORG}/_apis/projects"
    params = {"api-version": "7.1"}
    response = requests.get(url, auth=auth, params=params)
    
    if response.status_code != 200:
        st.error(f"Failed to fetch projects: {response.status_code} - {response.text}")
        st.stop()
    
    data = response.json()
    projects = [project["name"] for project in data.get("value", [])]
    
    if not projects:
        st.warning("No projects found or insufficient permissions.")
        st.stop()
    
    return projects

PROJECTS = fetch_projects()

def fetch_bugs(project):
    url = f"https://analytics.dev.azure.com/{ORG}/{project}/_odata/v3.0-preview/WorkItems?$filter=WorkItemType eq 'Bug'"
    response = requests.get(url, auth=auth)
    if response.status_code != 200:
        st.warning(f"Failed to fetch {project}: {response.status_code}")
        return project, None
    data = response.json()
    df = pd.json_normalize(data['value'])
    return project, df

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_drop = [col for col in df.columns if col.endswith("SK")]
    system_cols = ["AnalyticsUpdatedDate", "SystemRev", "SystemCreatedDate", "SystemChangedDate",
                   "AuthorizedDate", "StateChangeDate", "ActivatedDate", "ReactivateDate", "ResolvedDate"]
    cols_to_drop += system_cols
    cols_to_drop += [col for col in df.columns if col.startswith("Microsoft_VSTS_")]
    drop_if_exists = ["CommentCount", "Watermark", "Revision", "WorkItemType", "LeadTimeDays", "CycleTimeDays", "ChangedDate"]
    cols_to_drop += [c for c in drop_if_exists if c in df.columns]
    if "AssignedTo" in df.columns:
        cols_to_drop.append("AssignedTo")
    high_null_cols = df.columns[df.isnull().mean() > 0.90].tolist()
    cols_to_drop += high_null_cols
    df.drop(columns=set(cols_to_drop), inplace=True, errors="ignore")

    unified_module_col = "Custom_FeatureorModule"
    
    if "Custom_CategoryandModules" in df.columns:
        if unified_module_col in df.columns:
            df[unified_module_col] = df[unified_module_col].fillna(df["Custom_CategoryandModules"])
        else:
            df = df.rename(columns={"Custom_CategoryandModules": unified_module_col})
    
    categorical_cols = ["State", "Reason", "Severity", "Priority", unified_module_col,
                        "Custom_TestingPhaseList", "Custom_TestingType", "Custom_Platform", 
                        "Custom_Release", "Custom_TestType", "TagNames", "ValueArea", 
                        "Custom_Type", "ResolvedReason", "StateCategory"]
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    for col in categorical_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].replace({"nan": "Not Specified", "<NA>": "Not Specified", "None": "Not Specified"})
        df[col] = df[col].str.strip()
        df[col] = df[col].replace({"": "Not Specified"})
    if "Title" in df.columns:
        df["Title"] = df["Title"].fillna("")

    date_cols = ["CreatedDate", "InProgressDate", "CompletedDate", "ClosedDate"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if all(col in df.columns for col in ["InProgressDate", "CreatedDate"]):
        df["TimeToStart_Days"] = (df["InProgressDate"] - df["CreatedDate"]).dt.days
    if all(col in df.columns for col in ["CompletedDate", "CreatedDate"]):
        df["TimeToComplete_Days"] = (df["CompletedDate"] - df["CreatedDate"]).dt.days

    for col in ["TimeToStart_Days", "TimeToComplete_Days"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if pd.notnull(x) and x >= 0 else np.nan)

    return df

# ================================
# SYNTHETIC BUG GENERATION
# ================================


def generate_synthetic_bugs_for_cluster(cluster_df: pd.DataFrame, cluster_id: int, training_name: str, count: int = 4, focus_feature: str = None, focus_severity: str = None) -> list:
    feature_col = "Custom_FeatureorModule"  # unified name after preprocessing
    
    top_modules = ["Not Specified"]
    if feature_col in cluster_df.columns:
        counts = cluster_df[feature_col].value_counts()
        if not counts.empty:
            top_modules = counts.head(3).index.tolist()

    sample_titles = cluster_df['Title'].dropna().head(5).tolist()

    prompt = f"""
You are a senior QA engineer. Generate exactly {count} new plausible bug titles for this cluster.
Cluster: {training_name} (ID: {cluster_id}), {len(cluster_df)} real bugs
Common Modules: {', '.join(top_modules)}
Historical Examples:
""" + "\n".join([f"- {t}" for t in sample_titles]) + f"""

STRICT FOCUS RULES:
- ONLY generate bugs that are realistic for the module: **{focus_feature or 'any module'}**
"""

    if focus_severity:
        prompt += f"- Prioritize issues that would most likely be classified as **{focus_severity}** severity\n"

    prompt += """
- Focus on scale, concurrency, edge cases, regression, performance, security, integration, functional failures.
- Make sure the new bugs are realistic, relevant, and NON-OVERLAPPING with historical examples
- Avoid trivial variations — aim for interesting, plausible future defects
- Output ONLY a JSON object: {"titles": ["bug title 1", "bug title 2", ...]}
"""

    try:
        response = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8, 
            max_tokens=400,
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content.strip()

        json_match = re.search(r'(\{.*\})|(\[.*\])', raw_content, re.DOTALL)
        if json_match:
            clean_content = json_match.group(0)
        else:
            clean_content = raw_content

        parsed = pyjson.loads(clean_content)

        # Handle different potential formats (List vs Dict)
        if isinstance(parsed, dict):
            # Look for common keys if 'titles' isn't used
            titles = parsed.get("titles", parsed.get("bugs", list(parsed.values())[0]))
        else:
            titles = parsed

        # Validate we actually got strings
        if not isinstance(titles, list):
            raise ValueError("Parsed output is not a list")
            
        final_titles = [str(t).strip() for t in titles if t]

    except Exception as e:
        import streamlit as st # Assuming Streamlit based on your snippet
        st.warning(f"Cluster {cluster_id}: LLM parsing failed. Using fallbacks. Error: {str(e)[:50]}")
        
        # 4. Context-Aware Fallbacks
        # Instead of generic text, use the module names to make the fallback look real
        primary_mod = top_modules[0] if top_modules else "System"
        final_titles = [
            f"Unexpected {primary_mod} failure during high-concurrency stress test",
            f"Memory leak detected in {primary_mod} when processing large datasets",
            f"Intermittent race condition in {primary_mod} state synchronization",
            f"Boundary condition error in {primary_mod} validation logic"
        ]

    return final_titles[:count]  # Use the requested count

# ================================
# TRAINING & PREDICTION FUNCTIONS 
# ================================
@st.cache_resource(show_spinner="Training advanced AI models...")
def train_all_models(df: pd.DataFrame, training_name: str = "Combined", bugs_per_cluster: int = 4):
    if df.empty:
        return None

    df = df.copy()
    df["Title"] = df["Title"].fillna("").astype(str)
    df["Source"] = "Real"  # Mark as real

    with st.spinner("Generating semantic embeddings for real bugs..."):
        real_embeddings = embedder.encode(df["Title"].tolist(), show_progress_bar=False)
        np.save(MODEL_DIR / f"bug_embeddings_real_{training_name}.npy", real_embeddings)
        df.to_csv(MODEL_DIR / f"bug_metadata_{training_name}.csv", index=False)

    # --- Existing ML training (only on real data) ---
    num_cols = [c for c in ["TimeToStart_Days", "TimeToComplete_Days"] if c in df.columns]
    cat_cols = ["State", "Reason", "Severity", "Priority", "Custom_FeatureorModule", "Custom_CategoryandModules",
                        "Custom_TestingPhaseList", "Custom_TestingType", "Custom_Platform", "Custom_Release", "Custom_TestType",
                        "TagNames", "ValueArea", "Custom_Type", "ResolvedReason", "StateCategory"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    for c in cat_cols: df[c] = df[c].fillna("Unknown")
    for c in num_cols: df[c] = df[c].fillna(-1)

    label_encoders = {}
    X_cat_parts = []
    for c in cat_cols:
        le = LabelEncoder()
        encoded = le.fit_transform(df[c].astype(str))
        X_cat_parts.append(encoded.reshape(-1, 1))
        label_encoders[c] = le
    joblib.dump(label_encoders, MODEL_DIR / f"label_encoders_{training_name}.joblib")

    X_num = df[num_cols].values if num_cols else np.zeros((len(df), 0))
    X_cat = np.hstack(X_cat_parts) if X_cat_parts else np.zeros((len(df), 0))
    X = np.hstack([real_embeddings, X_num, X_cat])

    results = {
        "name": training_name,
        "n_bugs": len(df),
        "n_features": X.shape[1],
        "models": [],
        "cluster_prompts": None,
        "synthetic_df": None,
        "all_embeddings": None,
        "all_df": None
    }

    # Severity & Module models (unchanged)
        # Severity Model - Safe stratified split
    if "Severity" in df.columns:
        with st.spinner("Training Severity Prediction Model..."):
            df["Severity_Clean"] = df["Severity"].astype(str).str.replace(r"^\s*\d+\s*[-:]?\s*", "", regex=True).str.strip()
            le_sev = LabelEncoder()
            y = le_sev.fit_transform(df["Severity_Clean"])
            joblib.dump(le_sev, MODEL_DIR / f"le_severity_{training_name}.joblib")

            # Safe split: fall back to non-stratified if any class has <2 samples
            if len(df) > 1 and len(np.unique(y)) > 1:
                try:
                    # Check if every class has at least 2 samples
                    if np.min(np.bincount(y)) >= 2:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # non-stratified
                except:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                X_train, y_train = X, y

            model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            joblib.dump(model, MODEL_DIR / f"severity_model_{training_name}.joblib")
            results["models"].append("Severity Model → Saved")

    # Module Model - Same safe logic
    # Module Model - Now supports both Custom_FeatureorModule AND Custom_CategoryandModules
    if "Custom_FeatureorModule" in df.columns or "Custom_CategoryandModules" in df.columns:
        with st.spinner("Training Unified Module Prediction Model..."):
            # Create unified target column
            module_col = "Custom_FeatureorModule"
            df[module_col] = df[module_col] if module_col in df.columns else None
            
            if "Custom_CategoryandModules" in df.columns:
                # Fill main column with alternate where missing
                df[module_col] = df[module_col].fillna(df["Custom_CategoryandModules"])
            
            # Final cleanup: fill remaining missing with "Not Specified"
            df[module_col] = df[module_col].fillna("Not Specified").str.strip()
            df[module_col] = df[module_col].replace({"": "Not Specified"})

            # Now train on the unified column
            le_mod = LabelEncoder()
            y = le_mod.fit_transform(df[module_col].astype(str))
            joblib.dump(le_mod, MODEL_DIR / f"le_module_unified_{training_name}.joblib")

            # Safe train/test split
            if len(df) > 1 and len(np.unique(y)) > 1:
                try:
                    if np.min(np.bincount(y)) >= 2:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                except:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                X_train, y_train = X, y

            model = RandomForestClassifier(
                n_estimators=300,           # Slightly higher for better generalization
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=1
            )
            model.fit(X_train, y_train)
            joblib.dump(model, MODEL_DIR / f"module_model_unified_{training_name}.joblib")
            
            # Optional: Show top predicted modules
            top_modules_trained = pd.Series(le_mod.inverse_transform(y)).value_counts().head(10)
            # st.success(f"Module Model → Trained on {len(np.unique(y))} unified modules (e.g., {', '.join(top_modules_trained.index[:5].tolist())})")
            results["models"].append("Unified Module Model → Saved")

    # --- Clustering (only real bugs) ---
    df = df.copy()
    feature_col = "Custom_FeatureorModule"
    
    if "Custom_CategoryandModules" in df.columns:
        # Merge both columns: prioritize Custom_FeatureorModule, fallback to Custom_CategoryandModules
        df[feature_col] = df[feature_col].fillna(df["Custom_CategoryandModules"])
        df[feature_col] = df[feature_col].fillna("Not Specified").str.strip()
    else:
        # Ensure main column exists and is cleaned
        df[feature_col] = df[feature_col].fillna("Not Specified").str.strip()

    # Clustering (only real bugs) - NOW with unified modules
    with st.spinner("Clustering real bugs semantically (unified modules across projects)..."):
        n_clusters = min(8, max(1, len(df)//10))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['BugCluster'] = kmeans.fit_predict(real_embeddings)

        prompts = []
        for cid in sorted(df['BugCluster'].unique()):
            cluster = df[df['BugCluster'] == cid]
            # Now uses UNIFIED feature_col - works for BOTH project types
            top_modules = cluster[feature_col].value_counts().head(3).to_dict()
            titles = cluster['Title'].head(5).tolist()

            prompt = f"""Cluster #{cid} – {training_name} ({len(cluster)} bugs)
Common Modules: {json.dumps(top_modules, indent=2)}
Sample Bugs:
""" + "\n".join([f"- {t[:100]}..." for t in titles]) + """

Generate 3 high-value test cases to prevent this type of failure."""
            prompts.append({"cluster_id": int(cid), "prompt": prompt})

        prompt_df = pd.DataFrame(prompts)
        prompt_df.to_csv(OUT_DIR / f"cluster_prompts_{training_name}.csv", index=False)
        results["cluster_prompts"] = prompt_df
        results["unified_feature_col"] = feature_col  # Save for downstream use

    # --- Generate Synthetic Bugs per Cluster (now with unified top modules) ---
    with st.spinner("Generating hypothetical future bugs per cluster (using unified modules)..."):
        synthetic_titles = []
        for cid in sorted(df['BugCluster'].unique()):
            cluster_df = df[df['BugCluster'] == cid]
            # Pass unified feature column data to synthetic generator
            new_titles = generate_synthetic_bugs_for_cluster(cluster_df, cid, training_name, count=bugs_per_cluster)
            synthetic_titles.extend(new_titles)

        synthetic_df = pd.DataFrame({
            "Title": synthetic_titles,
            "Source": "AI-Predicted",
            "BugCluster": -1  # not clustered with real
        })
        if not synthetic_df.empty:
            # Add unified module info to synthetic bugs for better similarity search
            synthetic_df[feature_col] = "AI-Predicted (Multi-Cluster)"
            synthetic_df.to_csv(OUT_DIR / f"synthetic_bugs_{training_name}.csv", index=False)

        results["synthetic_df"] = synthetic_df

        # Compute embeddings for synthetic bugs
        if not synthetic_df.empty:
            synth_embeddings = embedder.encode(synthetic_df["Title"].tolist(), show_progress_bar=False)
            all_embeddings = np.vstack([real_embeddings, synth_embeddings])
            all_df = pd.concat([df, synthetic_df], ignore_index=True)
        else:
            all_embeddings = real_embeddings
            all_df = df.copy()

        results["all_embeddings"] = all_embeddings
        results["all_df"] = all_df

    return results
@st.cache_data
def prepare_embeddings(_df: pd.DataFrame):
    titles = _df["Title"].fillna("No title").tolist()
    return embedder.encode(titles, show_progress_bar=False)

# ================================
# UPDATED PREDICTION PROMPT (Hybrid)
# ================================
def generate_predictive_risk_prompt(feature_name: str, all_df: pd.DataFrame, all_embeddings: np.ndarray, top_k: int = 5) -> str:
    if all_df.empty or "Title" not in all_df.columns:
        return "No bug data available."

    query_vec = embedder.encode([feature_name])
    sims = cosine_similarity(query_vec, all_embeddings)[0]
    top_indices = sims.argsort()[-top_k*2:][::-1]  # Get more to split real/synthetic

    real_bugs = []
    synth_bugs = []

    for idx in top_indices:
        similarity = sims[idx]
        if similarity < 0.2:
            continue
        title = all_df.iloc[idx]["Title"]
        source = all_df.iloc[idx].get("Source", "Real")
        label = "🔴 Real" if source == "Real" else "🟡 Hypothetical (AI-predicted)"
        if source == "Real":
            real_bugs.append(f"- {title} (sim: {similarity:.2f}) {label}")
        else:
            synth_bugs.append(f"- {title} (sim: {similarity:.2f}) {label}")

    real_text = "\n".join(real_bugs[:top_k]) if real_bugs else "None with high similarity."
    synth_text = "\n".join(synth_bugs[:top_k]) if synth_bugs else "None generated yet."

    prompt = f"""
*** ROLE: SENIOR QA ARCHITECT & DEFECT PREDICTION SPECIALIST ***

FEATURE UNDER TEST:
"{feature_name}"

HISTORICAL BUGS (Real incidents from production):
{real_text}

HYPOTHETICAL RISKS (Previously predicted by AI for similar patterns):
{synth_text}

ANALYSIS INSTRUCTIONS:
- Use BOTH real and hypothetical bugs to deeply understand failure patterns.
- Abstract root causes: concurrency, state, validation, integration, scale, edge cases.
- Predict ENTIRELY NEW defects that are not in either list.
- Focus on latent risks that appear only at scale, under load, or in future scenarios.
- Dont repeat known bugs or trivial variations. They should not be redundant.
- Think like a QA architect anticipating future failures.
- They should be realistic and relevant to the feature mentioned. Should include various possible failure modes related to the feature.
- They should be most frequently occured issues, most well-known potential issues that can occur related to the feature mentioned.

STRICT RULES:
- ❌ NEVER repeat, paraphrase, or slightly reword any bug from above lists
- ✅ All predictions must be genuinely new and forward-looking

DELIVERABLE (JSON array only):
[
  {{
    "Predicted_Bug": "Concise new defect title",
    "Root_Cause_Pattern": "Abstract pattern inferred",
    "Why_This_Is_New": "Why not in real or hypothetical history",
    "Risk_Level": "High | Medium | Low",
    "Testing_Technique": "e.g., Stress, Boundary, Chaos Engineering"
    "Steps_to_Reproduce": "A detailed, step-by-step sequence of actions required to reproduce the issue, including preconditions, environment setup, test data, user inputs, and any specific conditions under which the issue occurs."
]
"""
    return prompt.strip()

def get_grok_predictions(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": "You are an expert QA architect skilled in predictive defect analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=8000,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq API Error: {str(e)}"

# ================================
# UI (unchanged except Tab 2 & 3 use new data)
# ================================
st.set_page_config(page_title="AI Bug Intelligence Platform", layout="wide")

# [Your beautiful CSS remains exactly the same]
st.markdown("""
<style>
    /* ============================================= */
    /* General layout & typography                   */
    /* ============================================= */
    .header-container h1 {
        color: #1e88e5;
        text-align: center;
        font-size: 2.8rem;
        margin-bottom: 0.3rem;
    }
    .tagline {
        text-align: center;
        color: #546e7a;
        font-size: 1.25rem;
        margin-top: 0;
    }

    /* ============================================= */
    /* PRIMARY ACCENT - light blue family            */
    /* Used for buttons + selected multiselect pills */
    /* ============================================= */
    /* Primary buttons */
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    div.stButton > button[kind="primary"],
    .stButton > button {
        background-color: #a3d8ff !important;
        border: 1px solid #81d4fa !important;
        color: #0d47a1 !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.3rem !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
    }

    button[kind="primary"]:hover,
    .stButton > button:hover {
        background-color: #90caf9 !important;
        border-color: #64b5f6 !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important;
    }

    button[kind="primary"]:active {
        background-color: #64b5f6 !important;
    }

    /* Multiselect selected items (pills/tags/chips) */
    .stMultiSelect [data-baseweb="tag"],
    span[data-baseweb="tag"] {
        background-color: #a3d8ff !important;
        color: #0d47a1 !important;
        border-radius: 12px !important;
        border: 1px solid #81d4fa !important;
        padding: 0.35rem 0.75rem !important;
        font-weight: 500 !important;
        margin: 0.2rem !important;
    }

    /* Hover on selected tag */
    .stMultiSelect [data-baseweb="tag"]:hover {
        background-color: #90caf9 !important;
        border-color: #64b5f6 !important;
    }

    /* Close button (×) inside tag */
    .stMultiSelect [data-baseweb="tag"] span[aria-label*="remove"] {
        color: #0d47a1 !important;
    }

    /* Multiselect dropdown items hover (optional consistency) */
    div[role="listbox"] li:hover {
        background-color: #e3f2fd !important;
    }

    /* ============================================= */
    /* Inputs / Text areas / Text inputs → white     */
    /* ============================================= */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput input,
    input[type="text"],
    input[type="password"],
    textarea,
    [data-testid="stMultiSelect"] input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #b0d4ff !important;
        border-radius: 6px !important;
        padding: 0.55rem 0.8rem !important;
    }

    /* Focus state */
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    [data-testid="stMultiSelect"] input:focus {
        border-color: #64b5f6 !important;
        box-shadow: 0 0 0 3px rgba(163, 216, 255, 0.35) !important;
    }

    /* Placeholder */
    ::placeholder {
        color: #90a4ae !important;
        opacity: 0.9 !important;
    }

    /* Labels */
    label {
        color: #263238 !important;
        font-weight: 500 !important;
    }

    /* Card-like containers (optional improvement) */
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    /* ────────────────────────────────────────────────
   FIX 1: Multiselect selected tags (project pills)
   Keep blue background, make text BLACK
──────────────────────────────────────────────── */
.stMultiSelect [data-baseweb="tag"],
span[data-baseweb="tag"] {
    background-color: #a3d8ff !important;
    color: #000000 !important;           /* ← changed from #0d47a1 to black */
    border-radius: 12px !important;
    border: 1px solid #81d4fa !important;
    padding: 0.35rem 0.75rem !important;
    font-weight: 500 !important;
    margin: 0.2rem !important;
}

/* Hover state - still looks good with black text */
.stMultiSelect [data-baseweb="tag"]:hover {
    background-color: #90caf9 !important;
    border-color: #64b5f6 !important;
    color: #000000 !important;
}

/* Close (×) button inside tag - make it dark too */
.stMultiSelect [data-baseweb="tag"] span[aria-label*="remove"] {
    color: #000000 !important;
}

/* ────────────────────────────────────────────────
   FIX 2: Make primary button TEXT black
   (affects both "Fetch Bugs" and "Start Bug Learning")
──────────────────────────────────────────────── */
button[kind="primary"],
button[data-testid="baseButton-primary"],
div.stButton > button[kind="primary"],
.stButton > button {
    background-color: #a3d8ff !important;
    border: 1px solid #81d4fa !important;
    color: #000000 !important;           /* ← changed from #0d47a1 to black */
    font-weight: 600 !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.3rem !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    transition: all 0.2s ease !important;
}

/* Hover - keep black text */
button[kind="primary"]:hover,
.stButton > button:hover {
    background-color: #90caf9 !important;
    border-color: #64b5f6 !important;
    color: #000000 !important;           /* black on hover too */
    box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important;
}

/* Active/pressed state */
button[kind="primary"]:active {
    background-color: #64b5f6 !important;
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📥 Fetch & Filter", "🧠 Bugs Learning", "🔮 Predict Potential Bugs"])

# TAB 1 - Unchanged
# with tab1:
#     st.markdown("<div class='card'><h2 style='color:#000000; font-weight:bold; margin-top:0'>Fetch Bug Data from Azure DevOps</h2></div>", unsafe_allow_html=True)
    
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         selected_projects = st.multiselect("**Select Project(s)**", PROJECTS, default=PROJECTS)
#     with col2:
#         data_mode = st.radio("**Project Selection Mode**", ["Multiple", "Single"], horizontal=True)

#     if st.button("🚀 Fetch Bugs", key="fetch", type="primary"):
#         if not selected_projects:
#             st.warning("Please select at least one project.")
#         else:
#             progress = st.progress(0)
#             results = {}
#             with ThreadPoolExecutor() as executor:
#                 futures = {executor.submit(fetch_bugs, p): p for p in selected_projects}
#                 for i, future in enumerate(as_completed(futures), 1):
#                     p, df = future.result()
#                     results[p] = df
#                     progress.progress(i / len(selected_projects))

#             processed = {}
#             all_dfs = []
#             for p, df in results.items():
#                 if df is not None and not df.empty:
#                     clean_df = preprocess_df(df)
#                     processed[p] = clean_df
#                     all_dfs.append(clean_df.assign(Project=p))

#                     with st.expander(f"**{p}** – {len(clean_df):,} bugs", expanded=False):
                        
#                             display_cols = ["WorkItemId", "Title", "Severity", "State"]
#                             if "WorkItemId" in clean_df.columns:
#                                 display_df = clean_df[display_cols].head(10).copy()
#                                 st.dataframe(display_df, width='stretch', hide_index=True)
#                             else:
#                                 st.dataframe(clean_df[["Title", "Severity", "State"]].head(10), width='stretch')
#                             csv = clean_df.to_csv(index=False).encode()
#                             st.download_button(f"📥 Download {p}", csv, f"{p}_bugs.csv", key=f"dl_{p}")

#             if all_dfs:
#                 combined_df = pd.concat(all_dfs, ignore_index=True)
#                 st.session_state.bug_data_combined = combined_df
#                 st.session_state.bug_data_individual = processed
#                 st.session_state.full_df = combined_df  # real only for fallback
#                 st.session_state.real_embeddings = prepare_embeddings(combined_df)
#                 st.success(f"✅ Successfully loaded **{len(combined_df):,}** total real bugs!")
#             else:
#                 st.error("No data fetched.")

with tab1:
    st.markdown("<div class='card'><h2 style='color:#000000; font-weight:bold; margin-top:0'>Fetch Bug Data from Azure DevOps</h2></div>", unsafe_allow_html=True)
    # ────────────────────────────────────────────────
# Inside tab1:
# ────────────────────────────────────────────────

    selected_projects = st.multiselect("**Select Project(s)**", PROJECTS, default=PROJECTS)
    
    # ── NEW: Display mode / filter mode ───────────────────────────────
    st.markdown("**What to show after fetching?**")
    view_mode = st.radio(
        "Display mode",
        options=[
            "All bugs (complete list)",
            "Only Blocker / Critical / Major",
            "Top 15 most recently created"
        ],
        index=0,  # default = show everything
        horizontal=True,
        key="view_mode_radio"
    )
    
    if st.button("🚀 Fetch Bugs", key="fetch", type="primary"):
        if not selected_projects:
            st.warning("Please select at least one project.")
        else:
            progress = st.progress(0)
            results = {}
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(fetch_bugs, p): p for p in selected_projects}
                for i, future in enumerate(as_completed(futures), 1):
                    p, df = future.result()
                    results[p] = df
                    progress.progress(i / len(selected_projects))
    
            processed = {}
            all_dfs = []           # complete real data (saved to session)
            display_dfs = []       # filtered view data (what user sees)
    
            for p, raw_df in results.items():
                if raw_df is None or raw_df.empty:
                    continue
    
                clean_df = preprocess_df(raw_df)
                clean_df = clean_df.assign(Project=p)
    
                # ──────────────── ALWAYS save full clean data ────────────────
                processed[p] = clean_df
                all_dfs.append(clean_df)
    
                # ──────────────── Create DISPLAY version (filtered) ────────────────
                display_df = clean_df.copy()
    
                if view_mode == "Only Blocker / Critical / Major":
                    severity_mask = display_df["Severity"].astype(str).str.contains(
                        r"(?i)(blocker|critical|major)", na=False
                    )
                    display_df = display_df[severity_mask]
    
                elif view_mode == "Top 15 most recently created":
                    if "CreatedDate" in display_df.columns:
                        display_df["CreatedDate"] = pd.to_datetime(display_df["CreatedDate"], errors="coerce")
                        display_df = display_df.sort_values("CreatedDate", ascending=False).head(15)
                    else:
                        display_df = display_df.head(15)  # fallback
    
                # else: "All bugs" → display_df remains = clean_df
    
                display_dfs.append(display_df)
    
                # Show expander with the *filtered* view
                count_shown = len(display_df)
                count_total = len(clean_df)
    
                label = f"**{p}** – showing {count_shown:,}"
                if count_shown < count_total:
                    label += f"  (of {count_total:,} total bugs)"
    
                with st.expander(label, expanded=False):
                    display_cols = ["WorkItemId", "Title", "Severity", "State"]
                    avail_cols = [c for c in display_cols if c in display_df.columns]
                    st.dataframe(
                    display_df[avail_cols],
                    use_container_width=True,
                    hide_index=True
                    )
    
                    # Download is ALWAYS full dataset
                    csv_full = clean_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📥 Download FULL {p} (all severities)",
                        data=csv_full,
                        file_name=f"{p}_bugs_full.csv",
                        mime="text/csv",
                        key=f"dl_full_{p}"
                    )
    
            # ── Save complete datasets to session state (for training etc.) ──
            if all_dfs:
                combined_full = pd.concat(all_dfs, ignore_index=True)
                st.session_state.bug_data_combined = combined_full
                st.session_state.bug_data_individual = processed
                st.session_state.full_df = combined_full
                st.session_state.real_embeddings = prepare_embeddings(combined_full)
    
                shown_count = sum(len(d) for d in display_dfs)
                total_count = len(combined_full)
    
                msg = f"✅ Loaded **{total_count:,}** total real bugs"
                if shown_count < total_count:
                    msg += f" — currently showing **{shown_count:,}** based on selected view mode"
    
                st.success(msg)
                st.session_state.fetch_completed = True
        
            else:
                st.error("No valid data fetched.")
                # ─────────────────────────────
    # Re-render previously fetched data after rerun
    # ─────────────────────────────
           




### heheh
with tab2:
    st.markdown("<div class='card'><h2 style='color:#000000; font-weight:bold; margin-top:0'>Bugs Learning</h2></div>", unsafe_allow_html=True)
    if "bug_data_combined" not in st.session_state:
        st.info("Please fetch and preprocess data in the first tab first.")
        st.stop()

    # Training scope selection
    training_scope = st.radio("**Learning Scope**", ["Combined (All Projects)", "Individual Project"], horizontal=True)

    if training_scope == "Individual Project":
        project_options = list(st.session_state.bug_data_individual.keys())
        selected_proj = st.selectbox("**Select Project**", project_options)
        df_to_use = st.session_state.bug_data_individual[selected_proj].copy()
        if "Project" not in df_to_use.columns:
            df_to_use["Project"] = selected_proj
        name = selected_proj.replace(" ", "_")
    else:
        df_to_use = st.session_state.bug_data_combined
        name = "Combined"

    st.markdown(
        f"<div style='text-align:center; font-size:1.3rem; color:#00E5FF; margin:1rem 0'>"
        f"**Training on:** {name} → {len(df_to_use):,} real bugs</div>",
        unsafe_allow_html=True
    )

    st.markdown("### Bug Learning and Prediction")
    st.markdown(
        "Click on the button below to use the **Historical Bug Data from Azure Devops** and train the model on data. "
        "This system analyzes all the past issues observed from the bugs and generates all possible likely bugs that"
        "could appear again based on **Historical Patterns**. "
    )
    bugs_per_cluster = st.slider(
        "Number of Bugs per Cluster",
        min_value=2,
        max_value=15,
        value=4,
        step=1,
        help="Higher = richer predictive coverage (e.g., 15 × 8 clusters = 120 total synthetic bugs)"
    )

    if st.button("Start Bug Learning", type="primary", key="start_training"):
        with st.spinner("Training models and generating AI-Predicted bugs..."):
            results = train_all_models(df_to_use, name, bugs_per_cluster=bugs_per_cluster)

        if results:
            # Save everything to session state so it persists
            st.session_state.training_results = results
            st.session_state.training_name = name
            st.session_state.training_df_to_use = df_to_use.copy()  # Save for later use in heatmap
            st.session_state.hybrid_df = results["all_df"]
            st.session_state.hybrid_embeddings = results["all_embeddings"]

            # Immediate success feedback
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Real Bugs", f"{results['n_bugs']:,}")
            with col2:
                st.metric("Synthetic Bugs", len(results.get("synthetic_df", [])))
            with col3:
                st.metric("Total Search Space", len(results["all_df"]))

            st.markdown("### Models Trained & Saved")
            for m in results["models"]:
                st.success(m)

            st.balloons()
            st.success("Hybrid AI training completed! Scroll down to explore the bug heatmap and details.")
            st.rerun()  # Refresh once to show the new section below

    # ========================
    # PERSISTENT RESULTS SECTION (Shown after training)
    # ========================
    if "training_results" in st.session_state:
        results = st.session_state.training_results
        name = st.session_state.training_name
        df_to_use = st.session_state.training_df_to_use  # Retrieve saved dataframe
        summary_df = df_to_use.copy()

        # Show synthetic bugs (same as before)
        if not results["synthetic_df"].empty:
            st.markdown(f"### AI-Predicted Bugs from model ({len(results['synthetic_df'])} total)")
            display_df = results["synthetic_df"][["Title", "Source"]].copy()
            st.dataframe(
                display_df.style.set_properties(**{'text-align': 'left', 'white-space': 'pre-wrap'}),
                width='stretch',
                height=min(600, len(display_df) * 40 + 100)
            )
            csv_synth = results["synthetic_df"].to_csv(index=False).encode()
            st.download_button(
                "Download AI-Predicted Bugs",
                csv_synth,
                f"ai_predicted_bugs_{name}.csv",
                "text/csv"
            )
        st.markdown("### Historical Bug Clusters & Test Case Prompts")
        st.markdown(
                "This section groups **historical bugs fetched from Azure DevOps** into clusters based on similar "
                "features and modules. By analyzing past issues, the system highlights **commonly occurring bug "
                "patterns** across different areas of the product. These clusters help identify modules that have "
                "historically experienced similar problems so that they can be used to "
                "proactively test those areas and anticipate potential future bugs."
            )
        st.dataframe(results["cluster_prompts"], width='stretch')
        csv_prompts = results["cluster_prompts"].to_csv(index=False).encode()
        st.download_button(
                "Download Test Case Prompts",
                csv_prompts,
                f"test_case_prompts_{name}.csv",
                "text/csv"
        )


    # ────────────────────────────────────────────────────────────────
    #   HEATMAP + BUG DETAILS FILTER
    # ────────────────────────────────────────────────────────────────
    if ("Custom_FeatureorModule" in df_to_use.columns or "Custom_CategoryandModules" in df_to_use.columns) and "Severity" in df_to_use.columns:
        st.markdown("#### Bug Heatmap by Feature & Severity")
        st.markdown(
            "<div style='font-size:14px; color:#444; margin-bottom:10px;'>"
            "This heatmap shows a detailed distribution of <b>historical bugs fetched from Azure DevOps</b>, "
            "grouped by <b>Feature/Module</b> and <b>Severity</b>. It helps highlight which system areas "
            "have historically experienced more issues and the severity levels associated with them."
            "</div>",
            unsafe_allow_html=True
        )

        summary_df = df_to_use.copy()
        # Clean Severity
        summary_df["Severity"] = summary_df["Severity"].astype(str).str.replace(r"^\s*\d+\s*[-:]?\s*", "", regex=True).str.strip()
        # Unify feature column
        feature_col = "Custom_FeatureorModule"
        if "Custom_CategoryandModules" in summary_df.columns:
            if feature_col in summary_df.columns:
                summary_df[feature_col] = summary_df[feature_col].fillna(summary_df["Custom_CategoryandModules"])
            else:
                summary_df = summary_df.rename(columns={"Custom_CategoryandModules": feature_col})
        summary_df[feature_col] = summary_df[feature_col].fillna("Not Specified").str.strip().replace({"": "Not Specified"})

        # Pivot table
        pivot = pd.pivot_table(
            summary_df,
            values='Title',
            index=feature_col,
            columns='Severity',
            aggfunc='count',
            fill_value=0,
            margins=True,
            margins_name="Total"
        )
        pivot = pivot.sort_values(by="Total", ascending=False)
        total_row = pivot.loc["Total"]
        pivot = pivot.drop("Total")
        pivot = pd.concat([pivot, pd.DataFrame([total_row])])
        display_pivot = pivot.astype(int)

        # Styling function
        def highlight_high(val):
            if isinstance(val, (int, float)) and val == 0:
                return 'color: black; font-weight: bold;'
            if isinstance(val, (int, float)) and val > 0:
                max_val = display_pivot.iloc[:-1, :-1].max().max() or 1
                intensity = min(val / max_val, 1.0)
                r = int(230 - intensity * 80)
                g = int(245 - intensity * 40)
                b = 255
                return f'background-color: rgb({r},{g},{b}); color: black; font-weight: bold; border-radius: 6px;'
            return ''

        styled_pivot = display_pivot.style \
            .map(highlight_high) \
            .format("{:,}") \
            .set_properties(**{'text-align': 'center', 'padding': '14px', 'font-size': '15px', 'border': '1px solid #444'}) \
            .set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#90D5FF'),
                    ('color', '#00E5FF'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('padding', '14px'),
                    ('font-size', '14px')
                ]},
                {'selector': 'td', 'props': [('min-width', '90px')]}
            ]) \
            .set_caption("Bug Heatmap")

        st.dataframe(styled_pivot, use_container_width=True, height=700)

        # ────────────────────────────────────────────────────────────────
        #   BUG DETAILS FILTER + FOCUSED SYNTHETIC GENERATION
        # ────────────────────────────────────────────────────────────────
        st.markdown("#### View Bug Details")
        if "bug_details_df" not in st.session_state:
            st.session_state.bug_details_df = None
            st.session_state.bug_details_info = None

        with st.form(key="bug_details_form"):
            col1, col2 = st.columns(2)
            with col1:
                feature_options = sorted(summary_df[feature_col].unique())
                selected_feature = st.selectbox(
                    "Select Feature/Module",
                    options=feature_options,
                    index=0,
                    key="detail_feature_tab2"
                )
            with col2:
                severity_options = sorted(summary_df["Severity"].unique())
                selected_severity = st.selectbox(
                    "Select Severity",
                    options=severity_options,
                    index=0,
                    key="detail_severity_tab2"
                )
            submit = st.form_submit_button("Show Bugs", type="primary")

            if submit:
                mask = (
                    (summary_df[feature_col] == selected_feature) &
                    (summary_df["Severity"] == selected_severity)
                )
                required_cols = ["WorkItemId", "Title", "State", "Priority", "CreatedDate", "Project"]
                available_cols = [c for c in required_cols if c in summary_df.columns]
                details_df = summary_df.loc[mask, available_cols].copy()

                if not details_df.empty:
                    if "CreatedDate" in details_df.columns:
                        details_df["CreatedDate"] = pd.to_datetime(details_df["CreatedDate"], errors="coerce")
                        details_df = details_df.sort_values("CreatedDate", ascending=False)
                        details_df["CreatedDate"] = details_df["CreatedDate"].dt.strftime("%Y-%m-%d")
                    display_df = details_df[["WorkItemId", "Title", "State", "Priority", "CreatedDate"]].copy()

                    st.session_state.bug_details_df = details_df
                    st.session_state.bug_details_info = (
                        f"Found {len(details_df)} bug(s) in **{selected_feature}** – Severity: **{selected_severity}**"
                    )
                    st.session_state.bug_details_display = display_df
                else:
                    st.session_state.bug_details_df = None
                    st.session_state.bug_details_info = "No bugs found for this combination."
                    st.session_state.bug_details_display = None

        # Results + download + new focused generation button
        if st.session_state.bug_details_info:
            if st.session_state.bug_details_df is not None:
                st.success(st.session_state.bug_details_info)
                st.dataframe(
                    st.session_state.bug_details_display,
                    use_container_width=True,
                    hide_index=True
                )

                download_cols = [
                    c for c in ["WorkItemId", "Title", "State", "Priority", "CreatedDate"]
                    if c in st.session_state.bug_details_df.columns
                ]
                csv = st.session_state.bug_details_df[download_cols].to_csv(index=False).encode()
                st.download_button(
                    label="Download These Bugs",
                    data=csv,
                    file_name=f"bugs_{selected_feature.replace(' ', '_')}_{selected_severity}.csv",
                    mime="text/csv",
                    key="download_filtered_bugs"
                )

                # ────────────────────────────────────────────────
                #   NEW: Focused synthetic bug generation
                # ────────────────────────────────────────────────
                st.markdown("---")
                st.info("You can now generate **additional AI-Predicted bugs focused only on this filtered view** "
                        "(feature + severity combination).")

                focused_count = st.slider(
                    "How many focused synthetic bugs to generate?",
                    min_value=3, max_value=20, value=8, step=1,
                    key="focused_synth_count"
                )

                if st.button("🧬 Generate Focused AI-Predicted Bugs for this Filter", type="primary"):
                    with st.spinner("Generating focused AI-Predicted bugs..."):
                        filtered_real = st.session_state.bug_details_df.copy()

                        new_titles = generate_synthetic_bugs_for_cluster(
                            cluster_df=filtered_real,
                            cluster_id=997,
                            training_name=f"Focused_{selected_feature}_{selected_severity}",
                            count=focused_count,
                            focus_feature=selected_feature,
                            focus_severity=selected_severity
                        )

                        if new_titles:
                            focused_df = pd.DataFrame({
                                "Title": new_titles,
                                "Source": "AI-Focused-Filter",
                                "BugCluster": -2,
                                feature_col: selected_feature,
                                "Severity": selected_severity,
                            })

                            # Append to global hybrid
                            if "hybrid_df" in st.session_state and "hybrid_embeddings" in st.session_state:
                                new_emb = embedder.encode(new_titles, show_progress_bar=False)

                                st.session_state.hybrid_df = pd.concat([
                                    st.session_state.hybrid_df,
                                    focused_df
                                ], ignore_index=True)

                                st.session_state.hybrid_embeddings = np.vstack([
                                    st.session_state.hybrid_embeddings,
                                    new_emb
                                ])

                            # ── Remember in focused history ──
                            if "focused_synthetic_latest" not in st.session_state:
                                st.session_state.focused_synthetic_latest = []

                            now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                            new_entries = [
                                {
                                    "Title": t,
                                    "Feature": selected_feature,
                                    "Severity": selected_severity,
                                    "GeneratedAt": now_str
                                }
                                for t in new_titles
                            ]
                            st.session_state.focused_synthetic_latest.extend(new_entries)

                            # Show immediate result
                            st.success(f"Added **{len(new_titles)} focused AI-Predicted bugs** for **{selected_feature}** ({selected_severity})")
                            st.markdown(f"**Just generated ({len(new_titles)} bugs):**")
                            st.dataframe(
                                pd.DataFrame(new_entries)[["Title", "Feature", "Severity", "GeneratedAt"]],
                                use_container_width=True,
                                hide_index=True
                            )

                            st.rerun()
                        else:
                            st.warning("No new titles were generated this time.")
                else:
                    st.info(st.session_state.bug_details_info)

        # Full heatmap download
        csv_data = display_pivot.to_csv().encode('utf-8')
        st.download_button(
            label="Download Full Bug Summary",
            data=csv_data,
            file_name=f"bug_heatmap_feature_severity_{name}.csv",
            mime="text/csv"
        )

    else:
        st.info("Required columns (Severity and Feature/Module) not available for bug heatmap.")


            # ────────────────────────────────────────────────────────────────
    #   FOCUSED SYNTHETIC BUGS HISTORY (persistent in this session)
    # ────────────────────────────────────────────────────────────────
    st.markdown("### Focused AI-Predicted Bugs (Generated per Filter)")
    st.markdown(
        "Use the filters above to generate **AI-predicted bugs derived from historical Azure DevOps bug data**. "
        "The system analyzes frequently occurring issues for the selected feature or module and predicts "
        "potential future bugs. For instance, selecting **Admin Panel** will generate predicted bugs based "
        "on past issues reported for the Admin Panel."
    )

    if "focused_synthetic_latest" not in st.session_state:
        st.session_state.focused_synthetic_latest = []

    if st.session_state.focused_synthetic_latest:
        focused_history_df = pd.DataFrame(st.session_state.focused_synthetic_latest)

        # Optional: group by feature + severity for better readability
        st.dataframe(
            focused_history_df[["Title", "Feature", "Severity", "GeneratedAt"]]
            .sort_values("GeneratedAt", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        # Download button for focused bugs
        csv_focused = focused_history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download All Focused Synthetic Bugs",
            data=csv_focused,
            file_name=f"focused_synthetic_bugs_{name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="download_focused_synthetic"
        )

        if st.button("Clear Focused Synthetic History", type="secondary"):
            st.session_state.focused_synthetic_latest = []
            st.rerun()

    else:
        st.info("No focused AI-Predicted bugs generated yet. Select a feature + severity filter and use the button above.")

        


# TAB 3 - Now uses hybrid data
with tab3:
    st.markdown("<div class='card'><h2 style='color:#000000; font-weight:bold; margin-top:0'>Predict New Potential Bugs</h2></div>", unsafe_allow_html=True)

    if "hybrid_df" not in st.session_state:
        st.info("👈 Please complete training in Tab 2 to enable hybrid prediction (real + synthetic risks).")
    else:
        all_df = st.session_state.hybrid_df
        all_embeddings = st.session_state.hybrid_embeddings

        st.markdown("### Describe the Feature Under Test")
        feature_desc = st.text_area("Describe the new feature or change you're testing",
                                    height=180,
                                    placeholder="e.g., Dealer uploads CNIC image → gets distorted on mobile, OCR fails in low light...",
                                    label_visibility="collapsed")

        # top_k = st.slider("**Number of Similar Bugs to Analyze (Real + Synthetic)**", 3, 12, 8)
        top_k = 20

        if st.button("🔍 Predict New Risks with Groq LLaMA (Hybrid)", type="primary"):
            if not feature_desc.strip():
                st.warning("Please describe the feature first.")
            else:
                prompt = generate_predictive_risk_prompt(feature_desc, all_df, all_embeddings, top_k)
                with st.expander("📜 Full Prompt Sent to Groq (Hybrid Context)", expanded=False):
                    st.code(prompt)

                with st.spinner("🧠 Groq LLaMA analyzing real + hypothetical patterns for deeper prediction..."):
                    response = get_grok_predictions(prompt)

                st.markdown("### 🤖 Predicted New & Hidden Risks (Beyond Real + Hypothetical)")
                try:
                    import json as pyjson
                    data = pyjson.loads(response)
                    if isinstance(data, list):
                        for i, item in enumerate(data, 1):
                            st.markdown(f"""
                            <div class='card'>
                                <h3 style='color:#00E5FF; margin-top:0'>🛑 Risk #{i}: {item.get('Predicted_Bug', 'Unknown')}</h3>
                                <p><strong>Root Cause Pattern:</strong> {item.get('Root_Cause_Pattern', 'N/A')}</p>
                                <p><strong>Why This Is New:</strong> {item.get('Why_This_Is_New', 'N/A')}</p>
                                <p><strong>Risk Level:</strong> {item.get('Risk_Level', 'N/A')}</p>
                                <p><strong>Testing Technique:</strong> {item.get('Testing_Technique', 'N/A')}</p>
                                <p><strong>Steps to Reproduce:</strong> {item.get('Steps_to_Reproduce', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.json(data)
                except Exception as e:
                    st.error("Failed to parse JSON response.")
                    st.markdown(response)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#88ffff; font-size:1.1rem'>"
            "Next-Gen Bug Intelligence • Hybrid Real + Synthetic Risk Modeling • Powered by Groq LLaMA</p>", 

            unsafe_allow_html=True)




















