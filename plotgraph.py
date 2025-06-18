import matplotlib.pyplot as plt
import json
import os

# Output path inside the 'out' folder
output_file = 'out/tetris_scores_linegraph.jpeg'

# Delete the old image file if it exists
if os.path.exists(output_file):
    os.remove(output_file)

# Load JSON data from file
with open('out/scores.json', 'r') as file:
    json_data = json.load(file)

# Extract lists for plotting
dates = [item["date"] for item in json_data]
scores = [item["score"] for item in json_data]

# Create the line plot
plt.plot(dates, scores, marker='o', linestyle='-', color='b')
plt.xlabel('date')
plt.ylabel('score')
plt.title('Tetris Scores über Zeit')

# Save the plot as a JPEG file inside 'out'
plt.savefig(output_file, format='jpeg')

# Show the plot
plt.show()

