import os

def remove_missing_images(txt_file_path, base_folder_path):
    with open(txt_file_path, 'r') as file:
        image_paths = file.readlines()
    
    image_paths = [path.strip() for path in image_paths]
    
    valid_image_paths = []
    for path in image_paths:
        full_image_path = os.path.join(base_folder_path, path)
        if os.path.exists(full_image_path):
            valid_image_paths.append(path)
    
    with open(txt_file_path, 'w') as file:
        for path in valid_image_paths:
            file.write(path + '\n')

    print("finish")


def list_image_files(folder_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    try:
        files = os.listdir(folder_path)
        image_files = [file for file in files if file.lower().endswith(image_extensions)]
        
        for image_file in image_files:
            print(image_file)
    
    except FileNotFoundError:
        print(f"폴더를 찾을 수 없습니다: {folder_path}")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

    

txt_file_path = '/home/jiwon/Dataset/FLIP_Dataset/MCIO/txt/celeb_fake_train.txt'
images_folder_path = '/home/jiwon/Dataset/FLIP_Dataset/MCIO/frame/'
remove_missing_images(txt_file_path, images_folder_path)
# list_image_files(images_folder_path)