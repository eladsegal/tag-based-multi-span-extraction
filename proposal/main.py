import json
from collections import defaultdict
import matplotlib.pyplot as plt

with open('drop_dataset/drop_dataset_dev.json') as f:
    data = json.loads(f.read())

multi_span_counts = defaultdict(int)
multi_span_count = 0
total_questions = 0
for key, entry in data.items():
    for qa_pair in entry['qa_pairs']:
        total_questions += 1
        span_count = len(qa_pair['answer']['spans'])
        if span_count > 1:
            multi_span_count += 1
            multi_span_counts[span_count] += 1

print("Multi-span Questions: " + str(multi_span_count))
print("Total Questions " + str(total_questions))
print(multi_span_counts)

fig, ax = plt.subplots(figsize=(10, 5))
x = [str(e) for e in sorted(list(multi_span_counts.keys()))]
y = [multi_span_counts[int(key)] for key in x]
plt.bar(x, y, width=0.2)
ax.set_xticks(x)
plt.xlabel('# of Spans in the Answer')
plt.ylabel('# of Questions')
plt.title('Dev Set')

# Make some labels.
rects = ax.patches
labels = [f'{multi_span_counts[int(x[i])]} ({round(multi_span_counts[int(x[i])] * 100 / total_questions, 2)}%)' for i in range(len(rects))]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
            ha='center', va='bottom')

fig.savefig('counts_dev.png')