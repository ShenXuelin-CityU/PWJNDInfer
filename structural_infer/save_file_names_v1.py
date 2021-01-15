#-*-coding-utf-8-*-

import os
import pandas as pd
import csv

def combine_csv_2(file1, file2):
    file1=open(file1)
    file2=open(file2)
    file3=csv.writer(open("./data/train.csv","w", newline=''))#合并后的文件
    for line1 in file1:
        for line2 in file2:
            Str1 = str(line1.rstrip('\r\n'))  #去掉回车
            Str2 = str(line2.rstrip('\r\n'))
            Str1 = Str1.split(',')  #分片成list
            Str2 = Str2.split(',')
            Str = Str1 + Str2[0:]
            # print(Str)
            file3.writerow(list(Str))  #写入
            break


def combine_csv_3(file1, file2, file3):
    file1=open(file1)
    file2=open(file2)
    file3=open(file3)
    file4=csv.writer(open("./data/train.csv","w", newline=''))#合并后的文件
    for line1 in file1:
        for line2 in file2:
            for line3 in file3:
                Str1 = str(line1.rstrip('\r\n'))  #去掉回车
                Str2 = str(line2.rstrip('\r\n'))
                Str3 = str(line3.rstrip('\r\n'))
                Str1 = Str1.split(',')  #分片成list
                Str2 = Str2.split(',')
                Str3 = Str3.split(',')
                Str = Str1 + Str2 + Str3[0:]
                # print(Str)
                file4.writerow(list(Str))  #写入
                break


def get_file_names(img_path1, img_path2, save_extention, dataset):
    if dataset == 'train':
        topen1 = open('./data/train1.csv', 'w')
        topen2 = open('./data/train2.csv', 'w')
        # fopen = open('train.csv', 'w')
    elif dataset == 'val':
        fopen = open('./data/val.csv', 'w')
    elif dataset == 'test':
        fopen = open('./data/test.csv', 'w')
    
    if save_extention:
        if dataset == 'train':
            for file in os.listdir(img_path1):
                string = '' + file + '\n'
                topen1.write(string)
            topen1.close()
            for file in os.listdir(img_path2):
                string = '' + file + '\n'
                topen2.write(string)
            topen2.close()
            combine_csv_2('./data/train1.csv', './data/train2.csv')
            # combine_csv_3('./data/train1.csv', './data/train2.csv', './data/train1.csv')
        elif dataset == 'val':
            for file in os.listdir(img_path1):
                string = '' + file + '\n'
                fopen.write(string)
            fopen.close()
        elif dataset == 'test':
            for file in os.listdir(img_path1):
                string = '' + file + '\n'
                fopen.write(string)
            fopen.close()
    else:
        if dataset == 'train':
            img_name_list1 = []
            img_name_list2 = []
            for file in os.listdir(img_path1):
                img_name_list1.append(file.split(".")[0])
            for file in img_name_list1:
                string = '' + file + '\n'
                topen1.write(string)
            topen1.close()
            for file in os.listdir(img_path2):
                img_name_list2.append(file.split(".")[0])
            for file in img_name_list2:
                string = '' + file + '\n'
                topen2.write(string)
            topen2.close()
            combine_csv('train1.csv', 'train2.csv')
        elif dataset == 'val':
            img_name_list = []
            for file in os.listdir(img_path1):
                img_name_list.append(file.split(".")[0])
            for file in img_name_list:
                string = '' + file + '\n'
                fopen.write(string)
            fopen.close()
        elif dataset == 'test':
            img_name_list = []
            for file in os.listdir(img_path1):
                img_name_list.append(file.split(".")[0])
            for file in img_name_list:
                string = '' + file + '\n'
                fopen.write(string)
            fopen.close()

if __name__ == "__main__":
    # img_path_train1 = './datajpg/raw'
    # img_path_train2 = './datajpg/exp'
    # img_path_val1 = './datajpg/val'
    # img_path_val2 = None
    # img_path_test1 = './datajpg/test'
    # img_path_test2 = None

    img_path_train1 = './data/OriginalPatch'
    img_path_train2 = './data/GroundPatch'
    # img_path_val1 = './data/val_ori'
    # img_path_val2 = None
    # img_path_test1 = './data/test'
    # img_path_test2 = None
    #
    train = 'train'
    val = 'val'
    test = 'test'
    
    get_file_names(img_path_train1, img_path_train2, True, train)
    # get_file_names(img_path_val1, img_path_val2, True, val)
    # get_file_names(img_path_test1, img_path_test2, True, test)
    
    


            
            