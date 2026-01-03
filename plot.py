import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from adjustText import adjust_text

# 1. Setup the Data
data = {
    'Industry': [
        'Energy', 'Materials', 'Capital Goods', 'Comm & Prof Services', 'Transportation',
        'Autos & Components', 'Consumer Durables', 'Consumer Services', 'Consumer Dist & Retail',
        'Food', 'Household Products', 'Health Care Equip', 'Pharma & Bio', 'Banks',
        'Financial Services', 'Insurance', 'Software & Services', 'Tech Hardware',
        'Semiconductors', 'Telecom', 'Media', 'Utilities', 'REITs', 'Real Estate Mgmt'
    ],
    'Valuation': [
        1.1849, 0.2263, -0.0383, -0.1327, -1.0742, -0.9777, -0.8480, -0.9196, -1.1435,
        -0.9944, -0.7151, -0.8189, 0.5057, -0.2165, -0.3130, -1.7869, 1.2727, -0.0225,
        0.8315, -0.9154, 0.8713, 0.7318, 1.5273, -1.0024
    ],
    'Momentum': [
        0.0991, 0.9067, 0.6779, -1.2396, -0.2544, 0.5561, -0.3976, -1.3355, -0.3514,
        -0.7194, -1.6222, -0.3987, 1.6846, 0.6159, -0.3042, -0.2438, -1.0020, 1.6810,
        2.7085, 0.0457, -0.1778, 0.3870, -0.5252, -0.1523
    ]
}

df = pd.DataFrame(data)

# 2. Define Ghibli-esque Style Constants
BG_COLOR = '#F7F1E3'      # "Rice Paper" / Cream
GRID_COLOR = '#8D99AE'    # Soft Blue-Grey
TEXT_COLOR = '#2B2D42'    # Dark Slate (Not pure black)
DOT_COLOR = '#EF233C'     # "Spirit Red" for contrast
ACCENT_COLOR = '#457B9D'  # "Totoro Blue"

# 3. Create the Plot
fig, ax = plt.subplots(figsize=(14, 10))

# Apply Backgrounds
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Add Quadrant Lines (The "Crosshair")
ax.axhline(0, color=ACCENT_COLOR, linewidth=1.5, alpha=0.6, linestyle='--')
ax.axvline(0, color=ACCENT_COLOR, linewidth=1.5, alpha=0.6, linestyle='--')

# Plot the Scatter points
scatter = ax.scatter(
    df['Valuation'], 
    df['Momentum'], 
    color=DOT_COLOR, 
    s=150,                # Size
    edgecolor=TEXT_COLOR, # Thin border to look hand-drawn
    linewidth=0.8,
    alpha=0.8,
    zorder=3
)

# 4. Typography and Labels (Serif for that bookish feel)
font_dict = {'family': 'serif', 'color': TEXT_COLOR, 'size': 10}
title_font = {'family': 'serif', 'color': TEXT_COLOR, 'size': 20, 'weight': 'bold'}

# Add Labels to points using adjustText to avoid overlap
texts = []
for i, txt in enumerate(df['Industry']):
    texts.append(ax.text(
        df['Valuation'][i], 
        df['Momentum'][i],
        txt,
        fontfamily='serif',
        fontsize=9,
        color=TEXT_COLOR,
        alpha=0.9
    ))

# Adjust text positions to avoid overlap
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

# 5. Final Polish
ax.set_title("Industry Groups: Valuation vs. Momentum", fontdict=title_font, pad=20)
ax.set_xlabel("Valuation (Z-Score)", fontdict=font_dict, labelpad=10)
ax.set_ylabel("Momentum (Z-Score)", fontdict=font_dict, labelpad=10)

# Customizing the Grid
ax.grid(True, which='both', color=GRID_COLOR, linestyle=':', linewidth=0.8, alpha=0.5)

# Remove top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(TEXT_COLOR)
ax.spines['left'].set_color(TEXT_COLOR)

plt.tight_layout()
plt.savefig('industry_valuation_momentum.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()