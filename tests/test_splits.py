import pandas as pd
from src.data.make_splits import make_patient_splits

def test_patient_is_only_in_one_split(tmp_path):
    path = tmp_path / "m.csv"; out = tmp_path / "s.csv"
    pd.DataFrame({"patient_id":["a","a","b","c","d","e"], "label_binary":[0,0,1,0,1,1], "quality_status":["accepted"]*6}).to_csv(path, index=False)
    result = make_patient_splits(str(path), str(out), 1)
    assert result.groupby("patient_id").split.nunique().max() == 1
