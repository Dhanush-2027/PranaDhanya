import os
import json
from pathlib import Path
import torch
import joblib

MODEL_DIR = Path('ai/models')
results = {}
for root, dirs, files in os.walk(MODEL_DIR):
    for f in files:
        path = Path(root)/f
        key = str(path.relative_to(MODEL_DIR))
        entry = {'path':str(path)}
        try:
            if path.suffix in ['.pt','.pth']:
                data = torch.load(path, map_location='cpu')
                # data may be dict or Tensor
                if isinstance(data, dict):
                    for k in ['best_accuracy','best_acc','accuracy','best_accuracy_epoch','best_val_acc']:
                        if k in data:
                            entry[k]=data[k]
                    # check nested
                    for k,v in data.items():
                        if isinstance(v,(int,float)) and 0<=v<=1 or (isinstance(v,(int,float)) and v>1):
                            pass
                results[key]=entry
            elif path.suffix in ['.joblib','.pkl']:
                obj = joblib.load(path)
                if hasattr(obj,'best_score_'):
                    entry['best_score_']=getattr(obj,'best_score_')
                if hasattr(obj,'score'):
                    entry['has_score_method']=True
                results[key]=entry
            else:
                results[key]=entry
        except Exception as e:
            entry['error']=str(e)
            results[key]=entry

print(json.dumps(results, indent=2))
