import matplotlib.pyplot as plt

from util import load_json

results = {}
results['Prosti metod'] = load_json(f'results/init_detection_roc/single.json')
results['Segmentacija do 3x3'] = load_json(f'results/init_detection_roc/3_segments.json')
results['Segmentacija do 4x4'] = load_json(f'results/init_detection_roc/4_segments.json')
results['Segmentacija do 5x5'] = load_json(f'results/init_detection_roc/5_segments.json')

plt.figure()
for key, val in results.items():
    plt.plot(val['fpr'], val['tpr'], label=key)
    
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("Stopa lažno pozitivnih")
plt.ylabel("Stopa stvarno pozitivnih")
plt.legend(loc="lower right")
plt.show()