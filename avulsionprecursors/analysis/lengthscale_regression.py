import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

file_path = 'data/manuscript_data/supplementary_table_1.csv'
df = pd.read_csv(file_path)

# plot dominant_wavelength vs variogram_range
plt.figure(figsize=(3, 2.5))
sns.regplot(x='dominant_wavelength', y='variogram_range', data=df, scatter_kws={'edgecolor': 'k'})

# Perform linear regression for dominant_wavelength vs variogram_range
slope, intercept, r_value, p_value, std_err = linregress(df['dominant_wavelength'], df['variogram_range'])

# add one to one line
plt.plot([0, 100], [0, 100], 'k--', linewidth=0.5)

# Annotate plot with regression equation, R^2, and p-value
plt.text(5, 65, f'$y = {slope:.2f}x {intercept:.2f}$\n$R^2 = {r_value**2:.2f}$\n$p = {p_value:.2e}$', 
         fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

# set tick labels and axis labels with font size 10-13 pt
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('$L_{{\lambda}}$ (km)', fontsize=13)
plt.ylabel('$L_C$ (km)', fontsize=13)

plt.tight_layout()
plt.show()
print('Dominant Wavelength')
print(df['dominant_wavelength'].describe())
print('Variogram Range')
print(df['variogram_range'].describe())

print('Dominant Wavelength / avg_width')
print((df['dominant_wavelength']/(df['avg_width']/1000)).describe())
print('Variogram Range / avg_width')
print((df['variogram_range']/(df['avg_width']/1000)).describe())

# plot median_meander_length vs variogram_range
sns.scatterplot(x='median_meander_length', y='variogram_range', data=df)
plt.plot([0, 50], [0, 50], 'k--', linewidth=0.5)

# Perform linear regression for median_meander_length vs variogram_range
slope_meander, intercept_meander, r_value_meander, p_value_meander, std_err_meander = linregress(df['median_meander_length'], df['variogram_range'])

# Print regression stats for meander length vs variogram range
print(f"Regression stats for meander length vs variogram range:")
print(f"Slope: {slope_meander:.2f}")
print(f"Intercept: {intercept_meander:.2f}")
print(f"R^2: {r_value_meander**2:.2f}")
print(f"P-value: {p_value_meander:.2e}")
print(f"Standard Error: {std_err_meander:.2f}")

plt.show()

ratios = df['variogram_range']/(df['median_meander_length']*2)

# Count ratios > 1 (larger than meander wavelength)
larger_than_meander = sum(r > 1 for r in ratios)
percent_larger = (larger_than_meander / len(ratios)) * 100

# Count ratios > 2 (more than double meander wavelength)
double_meander = sum(r > 2 for r in ratios)
percent_double = (double_meander / len(ratios)) * 100

print(f"Ratios larger than meander length: {larger_than_meander}/{len(ratios)} ({percent_larger:.1f}%)")
print(f"Ratios more than double meander length: {double_meander}/{len(ratios)} ({percent_double:.1f}%)")
print(ratios.describe())