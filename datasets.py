import os
from skimage import io
from skimage.transform import resize
from sklearn.utils import shuffle
import numpy as np

class datasets:
    def calculate(self, amount, percent):
        train = amount * percent / 100
        test = amount - train
        e =     {"train":round(train),
                "test":round(test)}
        return e
    def create(self, path, *size):
        folders = os.listdir(path)
        if "yes" in folders and "no" in folders:
            x_train_set_orig = []
            y_train_set = []
            x_test_set_orig = []
            y_test_set = []
            print("[*] Folder 'yes' and 'no' found") 
            for folder in folders:
                folder_path = os.path.join(path, folder)
                if os.path.isdir(folder_path):
                    if folder=="yes" or folder=="no":
                        folder_content = os.listdir(folder_path)
                        e = self.calculate(len(folder_content), 65)
                        train = e["train"]
                        test = e["test"]
                        print("\n[*] Folder:\t", folder)
                        print("[*] Images:\t", len(folder_content))
                        print("[*] Training:\t", train)
                        print("[*] Test:\t", test)
                        for img in folder_content:
                            img_path = os.path.join(folder_path, img)
                            array_img = io.imread(img_path)
                            image = resize(array_img, size[0],anti_aliasing=False, preserve_range=True)
                            if folder_content.index(img) < train:
                                x_train_set_orig.append(image)
                                if folder == "yes":
                                    y_train_set.append(1)
                                else:
                                    y_train_set.append(0)
                            else:
                                x_test_set_orig.append(image)
                                if folder == "yes":
                                    y_test_set.append(1)
                                else:
                                    y_test_set.append(0)                               
                    else:
                        print("[*] Folder", folder, "ignored")
            print("\n[*] Successfully generated dataset!")    
        else:
            raise Exception("[!] No folder 'yes' or 'no' found")  

        x_train_set_orig = np.array(x_train_set_orig)
        x_test_set_orig = np.array(x_test_set_orig)
        y_train_set = np.array([y_train_set])
        y_test_set = np.asarray([y_test_set])

        x_train_set_orig, y_train_set = shuffle(x_train_set_orig, y_train_set.T)
        x_test_set_orig, y_test_set = shuffle(x_test_set_orig, y_test_set.T)

        x_train_set_flat = x_train_set_orig.reshape(x_train_set_orig.shape[0],-1).T
        x_test_set_flat = x_test_set_orig.reshape(x_test_set_orig.shape[0],-1).T

        x_train_set = x_train_set_flat/255
        x_test_set = x_test_set_flat/255
    
        cds = {"x_train_set":x_train_set, 
            "x_test_set":x_test_set,
            "y_train_set":y_train_set.T,
            "y_test_set":y_test_set.T}
        return cds