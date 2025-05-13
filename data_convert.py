import os
import pandas as pd

base_original = r"C:\Users\Admin\Desktop\codes\CV_project_fill\LLM\app\data\original"
base_summary = r"C:\Users\Admin\Desktop\codes\CV_project_fill\LLM\app\data\summary"

data = []

for i in range(1, 301):
    cluster_name = f"Cluster_{i:03d}"
    original_folder = os.path.join(base_original, cluster_name, "original")
    summary_folder = os.path.join(base_summary, cluster_name)

    if not os.path.isdir(original_folder) or not os.path.isdir(summary_folder):
        continue

    # Ghép tất cả bài báo trong cluster thành 1 văn bản
    docs = []
    for filename in os.listdir(original_folder):
        file_path = os.path.join(original_folder, filename)
        if filename.endswith(".txt") and os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                docs.append(f.read().strip())
    if not docs:
        continue

    full_document = "\n".join(docs)

    # Dùng 1 trong 2 file tóm tắt (hoặc ghép cả 2)
    summaries = []
    for file in os.listdir(summary_folder):
        if file.endswith(".txt"):
            with open(os.path.join(summary_folder, file), 'r', encoding='utf-8') as f:
                summaries.append(f.read().strip())

    if not summaries:
        continue

    final_summary = summaries[0]  # hoặc " ".join(summaries) nếu muốn ghép cả 2

    data.append({"document": full_document, "summary": final_summary})

# Lưu lại dưới dạng CSV
df = pd.DataFrame(data)
df.to_csv("vit5_train_data.csv", index=False, encoding='utf-8')
print(f"Đã tạo {len(df)} mẫu.")
