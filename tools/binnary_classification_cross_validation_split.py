from PIL import Image
from zipfile import ZipFile
import os
import random

random.seed(42)

set1 = {"eMF": 20, "E": 41}
set2 = {"eMF": 19, "E": 42}
set3 = {"eMF": 20, "E": 42}


eMF = []
E = []
allfiles = []

for filename in os.listdir("."):
    if filename.endswith(".png"):
        allfiles.append(filename)
        if filename.startswith("resizedE"):
            E.append(filename.split("E")[1].split(".png")[0])
        elif filename.startswith("resizedeMF"):
            eMF.append(filename.split("eMF_")[1].split(".png")[0])


train_names_eMF = []
validation_names_eMF = []
test_names_eMF = []
test_names_eMF2 = []
test_names_eMF3 = []


train_names_E = []
validation_names_E = []
test_names_E = []
test_names_E2 = []
test_names_E3 = []


# Set 1
while len(train_names_eMF) < set1['eMF']:
    pop_indices = []
    while True:
        i = random.randint(0, len(eMF)-1)
        number = eMF[i].split('mrxs')[0]
        pop_indices = list(map(eMF.index, filter(lambda x: x.startswith(number), eMF)))
        if len(train_names_eMF) + len(pop_indices) <= set1['eMF']:
            break
    for step, i in enumerate(pop_indices):
        train_names_eMF.append(eMF.pop(i-step))

while len(train_names_E) < set1['E']:
    pop_indices = []
    while True:
        i = random.randint(0, len(E)-1)
        number = E[i].split('mrxs')[0]
        pop_indices = list(map(E.index, filter(lambda x: x.startswith(number), E)))
        if len(train_names_E) + len(pop_indices) <= set1['E']:
            break
    for step, i in enumerate(pop_indices):
        train_names_E.append(E.pop(i-step))

# Set 2
while len(validation_names_eMF) < set2['eMF']:
    pop_indices = []
    while True:
        i = random.randint(0, len(eMF)-1)
        number = eMF[i].split('mrxs')[0]
        pop_indices = list(map(eMF.index, filter(lambda x: x.startswith(number), eMF)))
        if len(validation_names_eMF) + len(pop_indices) <= set2['eMF']:
            break
    for step, i in enumerate(pop_indices):
        validation_names_eMF.append(eMF.pop(i-step))


while len(validation_names_E) < set2['E']:
    pop_indices = []
    while True:
        i = random.randint(0, len(E)-1)
        number = E[i].split('mrxs')[0]
        pop_indices = list(map(E.index, filter(lambda x: x.startswith(number), E)))
        if len(validation_names_E) + len(pop_indices) <= set2['E']:
            break
    for step, i in enumerate(pop_indices):
        validation_names_E.append(E.pop(i-step))


# Set 3
while len(test_names_eMF) < set2['eMF']:
    pop_indices = []
    while True:
        i = random.randint(0, len(eMF)-1)
        number = eMF[i].split('mrxs')[0]
        pop_indices = list(map(eMF.index, filter(lambda x: x.startswith(number), eMF)))
        if len(test_names_eMF) + len(pop_indices) <= set2['eMF']:
            break
    for step, i in enumerate(pop_indices):
        test_names_eMF.append(eMF.pop(i-step))


while len(test_names_E) < set2['E']:
    pop_indices = []
    while True:
        i = random.randint(0, len(E)-1)
        number = E[i].split('mrxs')[0]
        pop_indices = list(map(E.index, filter(lambda x: x.startswith(number), E)))
        if len(test_names_E) + len(pop_indices) <= set2['E']:
            break
    for step, i in enumerate(pop_indices):
        test_names_E.append(E.pop(i-step))


# Set 4
while len(test_names_eMF2) < set2['eMF']:
    pop_indices = []
    while True:
        i = random.randint(0, len(eMF)-1)
        number = eMF[i].split('mrxs')[0]
        pop_indices = list(map(eMF.index, filter(lambda x: x.startswith(number), eMF)))
        if len(test_names_eMF2) + len(pop_indices) <= set3['eMF']:
            break
    for step, i in enumerate(pop_indices):
        test_names_eMF2.append(eMF.pop(i-step))


while len(test_names_E2) < set2['E']:
    pop_indices = []
    while True:
        i = random.randint(0, len(E)-1)
        number = E[i].split('mrxs')[0]
        pop_indices = list(map(E.index, filter(lambda x: x.startswith(number), E)))
        if len(test_names_E2) + len(pop_indices) <= set3['E']:
            break
    for step, i in enumerate(pop_indices):
        test_names_E2.append(E.pop(i-step))



test_names_eMF3 = eMF 
test_names_E3 = E


#eMF
for file in train_names_eMF:
    filename = "predictioneMF" + "".join(file.split("_"))
    for name in allfiles:
        if name.startswith(filename):
            filename = name
            break
    os.replace(filename, "../set1/" + filename)

    filename = "resizedeMF_"+ file
    for name in allfiles:
        if name.startswith(filename):
            filename = name
            break
    os.replace(filename, "../set1/" + filename)

for file in validation_names_eMF:
    filename = "predictioneMF" + "".join(file.split("_"))
    for name in allfiles:
        if name.startswith(filename):
            filename = name
            break
    os.replace(filename, "../set2/" + filename)

    filename = "resizedeMF_"+ file
    for name in allfiles:
        if name.startswith(filename):
            filename = name
            break
    os.replace(filename, "../set2/" + filename)

for file in test_names_eMF:
    filename = "predictioneMF" + "".join(file.split("_"))
    for name in allfiles:
        if name.startswith(filename):
            filename = name
            break
    os.replace(filename, "../set3/" + filename)

    filename = "resizedeMF_"+ file
    for name in allfiles:
        if name.startswith(filename):
            filename = name
            break
    os.replace(filename, "../set3/" + filename)

for file in test_names_eMF2:
    filename = "predictioneMF" + "".join(file.split("_"))
    for name in allfiles:
        if name.startswith(filename):
            filename = name
            break
    os.replace(filename, "../set4/" + filename)

    filename = "resizedeMF_"+ file
    for name in allfiles:
        if name.startswith(filename):
            filename = name
            break
    os.replace(filename, "../set4/" + filename)

for file in test_names_eMF3:
    filename = "predictioneMF" + "".join(file.split("_"))
    for name in allfiles:
        if name.startswith(filename):
            filename = name
            break
    os.replace(filename, "../set5/" + filename)

    filename = "resizedeMF_"+ file
    for name in allfiles:
        if name.startswith(filename):
            filename = name
            break
    os.replace(filename, "../set5/" + filename)

#E
for file in train_names_E:
    filename = "predictionE" + "".join(file.split("_"))
    for name in allfiles:
        if name.startswith(filename + "."):
            filename = name
            break
    os.replace(filename, "../set1/" + filename)

    filename = "resizedE"+ file
    for name in allfiles:
        if name.startswith(filename + "."):
            filename = name
            break
    os.replace(filename, "../set1/" + filename)

for file in validation_names_E:
    filename = "predictionE" + "".join(file.split("_"))
    for name in allfiles:
        if name.startswith(filename + "."):
            filename = name
            break
    os.replace(filename, "../set2/" + filename)

    filename = "resizedE"+ file
    for name in allfiles:
        if name.startswith(filename + "."):
            filename = name
            break
    os.replace(filename, "../set2/" + filename)

for file in test_names_E:
    filename = "predictionE" + "".join(file.split("_"))
    for name in allfiles:
        if name.startswith(filename + "."):
            filename = name
            break
    os.replace(filename, "../set3/" + filename)

    filename = "resizedE"+ file
    for name in allfiles:
        if name.startswith(filename + "."):
            filename = name
            break
    os.replace(filename, "../set3/" + filename)


for file in test_names_E2:
    filename = "predictionE" + "".join(file.split("_"))
    for name in allfiles:
        if name.startswith(filename + "."):
            filename = name
            break
    os.replace(filename, "../set4/" + filename)

    filename = "resizedE"+ file
    for name in allfiles:
        if name.startswith(filename + "."):
            filename = name
            break
    os.replace(filename, "../set4/" + filename)


for file in test_names_E3:
    filename = "predictionE" + "".join(file.split("_"))
    for name in allfiles:
        if name.startswith(filename + "."):
            filename = name
            break
    os.replace(filename, "../set5/" + filename)

    filename = "resizedE"+ file
    for name in allfiles:
        if name.startswith(filename + "."):
            filename = name
            break
    os.replace(filename, "../set5/" + filename)
