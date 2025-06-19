import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict

# Set location for output file
output_file = 'out/tetris_linegraph.jpeg'

# Clear old image file if it exists
if os.path.exists(output_file):
    os.remove(output_file)

# Load data from scores.json
with open('out/scores.json', 'r') as file:
    json_data = json.load(file)

# Extract relevant data
scores_by_name = defaultdict(list)
for item in json_data:
    scores_by_name[item["name"]].append(item)
dates = [item["date"] for item in json_data]
scores = [item["score"] for item in json_data]

for name, entries in scores_by_name.items():
    # Sort entries by date if needed
    entries.sort(key=lambda x: x["date"])
    dates = [entry["date"] for entry in entries]
    scores = [entry["score"] for entry in entries]
    plt.plot(dates, scores, marker='o', linestyle='-', label=name)


# Create graph
# plt.plot(dates, scores, marker='o', linestyle='-', color='b')
plt.xlabel('Datum')
plt.ylabel('Score')
plt.title('Tetris Scores über Zeit', loc='left')
plt.legend(title="Name")

# Save plot as image in target directory
plt.savefig(output_file, format='jpeg')

plt.show()
