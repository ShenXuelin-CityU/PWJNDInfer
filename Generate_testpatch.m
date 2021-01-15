%Get test paches:cropped from test images
addpath( [ cd '\tools' ] );
file_path='.\Test_images\';%The address of test images
img_path_list = dir(strcat(file_path,'*.tif'));
files_name =sort_nat({img_path_list.name});
Frame=length(img_path_list);

%[h,w]:test image size
h=1080;
w=1920;

patchsize=64;
%Overlaps among nearby patches
overlap_num=14;

%Path to generated test patches
out_dir1='.\structural_infer\data\test\';
%There are no ground truth for test images, just make a copy 
out_dir2='.\structural_infer\data\exp_all\';

Patch_each_raw=floor(h/(patchsize-overlap_num));
Patch_in_frame=floor(w/(patchsize-overlap_num))*floor(h/(patchsize-overlap_num));
for f=0:Frame-1
    numfrm=1;
    num_pic=num2str(f);
    suff='.tif';
    name_open=strcat(file_path,files_name{f+1});
    img=imread(name_open);
    Yorg=rgb2gray(img);
     for col=1:floor(w/(patchsize-overlap_num))
        for row=1:floor(h/(patchsize-overlap_num))
            patch=Yorg((row-1)*(patchsize-overlap_num)+1:(row-1)*(patchsize-overlap_num)+patchsize,(col-1)*(patchsize-overlap_num)+1:(col-1)*(patchsize-overlap_num)+patchsize);
            patchindex=f*Patch_in_frame+(col-1)*Patch_each_raw+row; 
            num=num2str(patchindex);
            suff2='.bmp';
            filename1=strcat(out_dir1,'Ori',num,suff2);   
            filename2=strcat(out_dir2,'Ori',num,suff2); 
            imwrite(uint8(patch),filename1,'bmp');
            imwrite(uint8(patch),filename2,'bmp');
        end
    end

    clear Yor;
end