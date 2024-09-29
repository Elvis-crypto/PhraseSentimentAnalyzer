# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:54:02 2024

@author: loriz
"""

import matplotlib.pyplot as plt

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 10))

# Detailed text for each step
steps_details = {
    "Data Collection": "Reddit API / Pushshift\nSearch for specific phrases\nFiltered by subreddits",
    "Text Preprocessing": "Clean and tokenize text\nMatch target phrases\nRemove stop words",
    "Sentiment Analysis": "VADER/TextBlob analysis\nFine-tuned models (e.g. BERT)\nMulti-class sentiment",
    "Context Understanding": "Aspect-based sentiment (ABSA)\nPhrase relevance\nNamed Entity Recognition (NER)",
    "Data Storage and Aggregation": "Store results in SQL/NoSQL\nAggregate by phrase, platform, and time",
    "Visualization and Reporting": "Dashboards (Plotly, Dash)\nSentiment trends over time\nAutomated alerts for changes"
}

# Define rectangles with more details inside
rects_detailed = {
    "Data Collection": {"xy": (0.1, 0.83), "width": 0.8, "height": 0.1, "color": "lightblue"},
    "Text Preprocessing": {"xy": (0.1, 0.68), "width": 0.8, "height": 0.1, "color": "lightgreen"},
    "Sentiment Analysis": {"xy": (0.1, 0.53), "width": 0.8, "height": 0.1, "color": "lightsalmon"},
    "Context Understanding": {"xy": (0.1, 0.38), "width": 0.8, "height": 0.1, "color": "lightcoral"},
    "Data Storage and Aggregation": {"xy": (0.1, 0.23), "width": 0.8, "height": 0.1, "color": "lightsteelblue"},
    "Visualization and Reporting": {"xy": (0.1, 0.08), "width": 0.8, "height": 0.1, "color": "lightgoldenrodyellow"},
}

# Draw rectangles with detailed steps
for label, rect_props in rects_detailed.items():
    ax.add_patch(plt.Rectangle(rect_props["xy"], rect_props["width"], rect_props["height"], 
                               color=rect_props["color"], edgecolor='black', linewidth=1.5))
    # Add detailed text for each step inside the boxes
    plt.text(rect_props["xy"][0] + 0.4, rect_props["xy"][1] + 0.05, steps_details[label], fontsize=11, ha="center", va="center")

# Adjust arrow size to avoid overlap
for i in range(5):
    plt.arrow(0.5, 0.83 - i*0.15, 0, -0.02, width=0.003, head_width=0.02, head_length=0.03, fc='black', ec='black')

# Set the plot limits and remove axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Show the improved diagram with detailed text and smaller arrows
plt.title("Architecture for Sentiment Analysis on Taste Enhancers", fontsize=14)
plt.show()
