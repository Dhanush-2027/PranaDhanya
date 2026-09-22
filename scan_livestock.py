import os

def count_files(dir_path):
    if not os.path.exists(dir_path):
        return 0
    cnt = 0
    for root, dirs, files in os.walk(dir_path):
        cnt += len(files)
    return cnt

print("Cattle:", count_files(r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\cattle_diseases"))
print("Dog:", count_files(r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\dog_skin_disease"))
print("Goat:", count_files(r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\livestock"))
