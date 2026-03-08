import json
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [np.nan, 2, np.nan]})
df_clean = df.replace(np.nan, None)
print(json.dumps(df_clean.to_dict(orient='records')))
