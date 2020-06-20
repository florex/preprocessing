import json
import numpy as np
with open("tf_idf.json", "r") as f :
    tf_idf = json.load(f)

n= 100
overlaps = {}
imp = {}
mat = np.zeros((10,10),dtype=np.int32)
for i in range(10) :
    i_most_n = [(k, v) for k, v in sorted(tf_idf[i].items(), key=lambda item: item[1])][::-1][:n]
    i_most_n = {k: v for (k, v) in i_most_n}
    imp.update({i:i_most_n})

for i in range(10) :
    for j in range(i+1,10) :
        for k in imp[i] :
            if k in imp[j] :
                mat[i][j] += 1


for i in range(10) :
    mat[i][i] = 100
    for j in range(i+1,10) :
        mat[j][i] = mat[i][j]

print(mat)