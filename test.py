import torch

def test():

	one = torch.ones((1, 3))
	for x in range(10):
		one = one.add(1)
		yield one


arr = []
for x in test():
	print(x)
	arr.append(x)
print(arr)





	