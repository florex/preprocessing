import json
with open("words_freq.json", "r") as f :
    freqs = json.load(f)
n= 10
for i in range(10) :
    i_most_n = [(k, v) for k, v in sorted(freqs[i].items(), key=lambda item: item[1])][::-1][:n]
    print("Class "+ str(i)+ " :   "+str(i_most_n))
