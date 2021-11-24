# import string
# f = open('./testing_2.txt', 'w')

# file = open("./training_set_4.txt", "r").readlines()

# neg = 0
# pos = 0
# for line in file:
# 	ele = line.split('\t')
# 	text = ele[0]
# 	label = ele[1].strip()

# 	text = str(text)
# 	label = str(label)

# 	new_item = text + label


# 	if label == "0" and neg != 2242:
# 		neg += 1
# 		f.write(text + "\t" + label + "\n")
# 	if label == "1":
# 		pos += 1
# 		f.write(text + "\t" + label + "\n")

# print("Pos counts: ", pos)
# print("Neg counts: ", neg)

# f.close()

# print("complete.")


	