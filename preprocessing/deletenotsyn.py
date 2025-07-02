import os
import pandas as pd

def sync_labels_and_images(image_dir, label_file):
    # Get all image files
    image_files = {f.split('.')[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))}
    
    # Read label file (assuming it has a column with image filenames)
    labels_df = pd.read_csv(label_file)
    label_files = set(labels_df['img'].str.split('.').str[0])  # Adjust 'image_name' to your column name

    # Find mismatches
    images_to_delete = image_files - label_files
    labels_to_delete = label_files - image_files

    # Delete unmatched images
    for img in images_to_delete:
        os.remove(os.path.join(image_dir, f"{img}.jpg"))  # Adjust extension as needed
        print(f"Deleted unmatched image: {img}.jpg")
    
    # Note: Deleting labels from CSV is more complex; consider filtering the CSV instead
    if labels_to_delete:
        filtered_df = labels_df[~labels_df['img'].str.split('.').str[0].isin(labels_to_delete)]
        filtered_df.to_csv(label_file, index=False)
        print(f"Updated {label_file} by removing unmatched labels.")

if __name__ == "__main__":
    image_directory = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\testset"
    label_file = "C:\\Users\\Iamnearu\\Documents\\ThucTapAI\\InternAI\\data\\label_cleaned.csv"
    sync_labels_and_images(image_directory, label_file)