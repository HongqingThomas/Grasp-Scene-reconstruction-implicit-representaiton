import csv
import matplotlib.pyplot as plt
# csv_file_path = "./GIGA/data/experiments/23-09-18-19-53-56/jointly_optimize.csv"
# csv_file_path = "./GIGA/data/experiments/23-09-18-19-56-17/jointly_optimize.csv"
# csv_file_path = "./GIGA/data/experiments/23-09-18-19-58-49/jointly_optimize.csv"
# csv_file_path = "./GIGA/data/experiments/23-09-18-20-01-16/jointly_optimize.csv"
csv_file_path = "./GIGA/data/experiments/23-09-18-20-03-42/jointly_optimize.csv"
data = []
loss_occ = []
succ = []
with open(csv_file_path, newline= '') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        if i == 0:
            i += 1
        else:
            # data.append(row)
            loss_occ.append(float(row[2]))
            succ.append(int(row[1]))

# print("loss_occ:", loss_occ)
# print("succ:", succ)

plt.scatter(loss_occ, succ, c=succ, cmap='coolwarm', marker = 'o')
plt.xlabel('loss_occ')
plt.ylabel('grasp success')
plt.title('relations between scene construction and grasp success rate in individual cases')
plt.savefig('fig/seed5.png')
plt.show()