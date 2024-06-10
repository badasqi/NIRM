import pandas as pd
import numpy as np

# Чтение CSV файла
df = pd.read_csv('./DMImageDetectionScreenshot-main/results_tst/COCO_results.csv')

# Указание столбца, который необходимо модифицировать
column_name = 'Grag2021_latent'

# Применение логического условия и замена значений
df[column_name] = np.where(df[column_name] > 0, 1, 0)

# Сохранение изменённого DataFrame в новый CSV файл
df.to_csv('./out_csv/COCO_test_out.csv', index=False)

print("Значения в столбце успешно изменены и сохранены")

# Вычисление точности (accuracy)
predicted_values = [0] * 50

accuracy = np.mean(np.array(predicted_values) == df[column_name].tolist())

print(f"Точность (accuracy): {accuracy * 100:.2f}%")

