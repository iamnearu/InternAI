from datasets import load_dataset

# Bước 1: Load dataset từ file CSV
dataset = load_dataset("csv", data_files=r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\dataset_donut\donut_labels.csv")

# Bước 2: Kiểm tra xem có đúng dữ liệu không
print(dataset)                # In ra thông tin tổng quát
print(dataset["train"][0])   # In thử một dòng dữ liệu
