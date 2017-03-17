atoms = ['C','c','H','h','N','n']
mol = "ch4"
count = 0
for index in range(0,len(mol)):
    if mol[index] in atoms:
        print mol[index]
        for index2 in range(index+1,len(mol)):
            if mol[index2].isdigit():
                count += index2
            else:
                count += 1
print count











