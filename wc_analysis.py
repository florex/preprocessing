import json
with open("wc_tf_idf.json", "r") as f :
    wc_dict = json.load(f)

min = 120000
max = 0

n100 = 0
n200 = 0;
n300 = 0;
n400 = 0;
n500 = 0;
growth = {0:0, 100:0,200:0,300:0,400:0,500:0,600:0,700:0,800:0,900:0,1000:0}
somme = 0;
for key, value in wc_dict.items() :
    if value < min :
        min = value

    if value > max :
        max = value

    for n in growth :
        if value >= n :
            growth[n] = growth[n]+1

    somme += value

avg = somme*1.0/len(wc_dict)

print("Effectifs cumulés : ",growth)
print("Moyenne du nombre de mots : ", avg)
print("Maximun du nombre de mots : ",max)
print("Minimum du nombre de mots : ",min)