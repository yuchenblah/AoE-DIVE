import json
from collections import Counter

with open('/data9/fengyuchen/AoE-DIVE/aoe_cases.json', 'r') as f:
    data = [json.loads(line.strip()) for line in f]
transposed_data = list(zip(*data))

most_frequent_experts = []
for column in transposed_data:
    flattened_column = [item for sublist in column for item in sublist]
    count = Counter(flattened_column)
    most_common = count.most_common(1)
    most_frequent_experts.append(most_common[0][0])

print(most_frequent_experts)