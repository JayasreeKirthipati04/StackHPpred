import os
import re
import math
import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import warnings
from transformers import AlbertModel, AlbertTokenizer

# Suppress annoying warnings for professional output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

# ==========================================
# 1. FEATURE EXTRACTION FUNCTIONS
# ==========================================

# 1.1 CTDD Feature Extraction
def Count1(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number += 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code

def CTDD(fastas):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN', 'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA', 'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG', 'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD', 'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY', 'polarizability': 'GASDT', 'charge': 'KR',
        'secondarystruct': 'EALMQKRH', 'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY', 'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV', 'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV', 'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG', 'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS', 'polarizability': 'CPNVEQIL', 'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT', 'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW', 'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI', 'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC', 'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC', 'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED', 'polarizability': 'KMHFRYW', 'charge': 'DE',
        'secondarystruct': 'GNPSD', 'solventaccess': 'MSPTHY'
    }

    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 
        'hydrophobicity_PONP930101', 'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 
        'hydrophobicity_FASG890101', 'normwaalsvolume', 'polarity', 'polarizability', 'charge', 
        'secondarystruct', 'solventaccess'
    )

    features = {}
    for name, sequence in fastas:
        code = []
        for p in property:
            code = code + Count1(group1[p], sequence) + Count1(group2[p], sequence) + Count1(group3[p], sequence)
        features[name] = np.array(code, dtype=np.float32)
    return features


# 1.2 KSCTriad Feature Extraction (Standard iFeature Implementation, kspace=3)
def KSCTriad(fastas, max_gap=3):
    AAGroup = {
        'g1': 'AGV', 'g2': 'ILFP', 'g3': 'YMTS',
        'g4': 'HNQW', 'g5': 'RK', 'g6': 'DE', 'g7': 'C'
    }
    myGroups = sorted(AAGroup.keys())
    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features_list = [g1 + '.' + g2 + '.' + g3 for g1 in myGroups for g2 in myGroups for g3 in myGroups]
    
    features = {}
    for name, sequence in fastas:
        full_code = []
        for gap in range(max_gap + 1): # Gaps 0, 1, 2, 3
            myDict = {f: 0 for f in features_list}
            
            for i in range(len(sequence)):
                if i + gap + 1 < len(sequence) and i + 2 * gap + 2 < len(sequence):
                    if sequence[i] in AADict and sequence[i + gap + 1] in AADict and sequence[i + 2 * gap + 2] in AADict:
                        fea = AADict[sequence[i]] + '.' + AADict[sequence[i + gap + 1]] + '.' + AADict[sequence[i + 2 * gap + 2]]
                        myDict[fea] += 1
                        
            values = list(myDict.values())
            maxValue, minValue = max(values), min(values)
            
            code = []
            for f in features_list:
                if maxValue == 0:
                    code.append(0.0)
                else:
                    code.append((myDict[f] - minValue) / maxValue)
                    
            full_code.extend(code)
            
        features[name] = np.array(full_code, dtype=np.float32)
        
    return features


# 1.3 PTAB (ProtTrans ALBERT BFD) Embeddings
def PTAB(fastas, model_name="Rostlab/prot_albert"):
    print(f"Loading PTAB Model ({model_name})...")
    
    # Automatically detect GPU, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device.type.upper()} for PTAB embeddings")
    
    # This automatically downloads the correct tokenizer and weights from HuggingFace
    tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AlbertModel.from_pretrained(model_name)
    model = model.to(device)
    model = model.eval() 
    
    features = {}
    for name, sequence in fastas:
        # ProtTrans expects spaces between amino acids and replaces rare AAs with X
        seq_spaced = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        inputs = tokenizer(seq_spaced, return_tensors='pt')
        
        # Move inputs to the selected device (GPU or CPU)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Remove CLS and SEP tokens
            hidden_states = outputs.last_hidden_state[0, 1:-1]
            # Mean pooling over the sequence length to get 1D embedding
            # Move back to CPU before converting to numpy
            embedding = torch.mean(hidden_states, dim=0).cpu().numpy()
            
        features[name] = np.array(embedding, dtype=np.float32)
    return features


# ==========================================
# 2. PREDICTION PIPELINE
# ==========================================

def read_fasta(file_path):
    fastas = []
    with open(file_path, 'r') as f:
        name, seq = "", ""
        for line in f:
            if line.startswith(">"):
                if name:
                    fastas.append((name, seq))
                name = line.strip()[1:]
                seq = ""
            else:
                seq += line.strip()
        if name:
            fastas.append((name, seq))
    return fastas

def load_pkl_model(filepath):
    # Apply joblib randomstate patch just in case (from your original snippet)
    import numpy as np
    def _randomstate_ctor_workaround(*args):
        return np.random.RandomState()
    if not hasattr(np.random, '__randomstate_ctor'):
        np.random.__randomstate_ctor = _randomstate_ctor_workaround
        
    return joblib.load(filepath)

def main():
    parser = argparse.ArgumentParser(description="StackHPpred: Meta-predictor for Halophilic Proteins")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    args = parser.parse_args()
    
    print(f"Reading FASTA file: {args.input}")
    fastas = read_fasta(args.input)
    if not fastas:
        print("No sequences found in FASTA.")
        return
    
    names = [f[0] for f in fastas]
    
    # 1. Compute Base Features
    print("\nComputing CTDD Features...")
    feat_ctdd = CTDD(fastas)
    
    print("Computing KSCTriad Features...")
    feat_ksctriad = KSCTriad(fastas)
    
    print("Computing PTAB Embeddings...")
    feat_ptab = PTAB(fastas)
    
    # 2. Run Base Models to get probabilities
    classifiers = ["AB", "ANN", "CB", "ERT", "GB", "LGB", "LRT", "RF", "SVM", "XGB"]
    
    meta_features = []
    print("\nGenerating Probabilities from 30 Base Models...")
    
    # We must iterate exactly in the order the meta model expects:
    # Top 3 features x 10 classifiers = 30 dimensions
    # Order: CTDD(10 clfs) -> KSCTriad(10 clfs) -> PTAB(10 clfs)
    
    for feat_name, feat_dict in zip(['CTDD', 'KSCTriad', 'PTAB'], [feat_ctdd, feat_ksctriad, feat_ptab]):
        X_base = np.array([feat_dict[name] for name in names])
        
        for clf in classifiers:
            model_path = os.path.join("models", "baseline_models", f"{clf}_{feat_name}.pkl")
            model = load_pkl_model(model_path)
            
            # If this specific model needed scaling, load its scaler
            scaler_path = os.path.join("models", "baseline_models", f"scaler_{clf}_{feat_name}.pkl")
            if os.path.exists(scaler_path):
                scaler = load_pkl_model(scaler_path)
                X_pred = scaler.transform(X_base)
            else:
                X_pred = X_base
                
            probs = model.predict_proba(X_pred)[:, 1]
            meta_features.append(probs)
            
    # Transpose so each row is a sequence, and each column is a feature probability (N x 30)
    X_meta = np.array(meta_features).T 
    
    # 3. Final Meta-Model Prediction
    print("\nRunning StackHPpred Meta-Model...")
    meta_scaler = load_pkl_model(os.path.join("models", "meta_scaler.pkl"))
    meta_model = load_pkl_model(os.path.join("models", "StackHPpred.pkl"))
    
    X_meta_scaled = meta_scaler.transform(X_meta)
    final_probs = meta_model.predict_proba(X_meta_scaled)[:, 1]
    
    # 4. Save Results
    results = []
    for i, name in enumerate(names):
        prob = final_probs[i]
        label = "HP" if prob >= 0.5 else "non-HP"
        results.append({"Sequence_Name": name, "Probability": round(prob, 4), "Prediction": label})
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(args.output, index=False)
    print(f"\n✅ Predictions successfully saved to {args.output}")

if __name__ == '__main__':
    main()
