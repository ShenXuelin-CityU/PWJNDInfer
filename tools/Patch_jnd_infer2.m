%----------------------------------------------------------------------
%Modified at 2021/01/10
function [stas_p,img_jnd] = Patch_jnd_infer2(img_inf,files_name,ori_patch_dir,ori_img_dir,file_path)
Frame=img_inf(1);
h=img_inf(2);
w=img_inf(3);
patchsize=img_inf(4);
overlap=img_inf(5);

img_JND=zeros(floor(h/(patchsize-overlap))*(patchsize-overlap)+overlap,floor(w/(patchsize-overlap))*(patchsize-overlap)+overlap);
stas_p=zeros(1,Frame);
img_jnd=cell(1,Frame);
patch_each_frame=floor(h/(patchsize-overlap))*floor(w/(patchsize-overlap));
%Ori_img name list
ori_path_list = dir(strcat(ori_img_dir,'*.png'));
ori_files_name =sort_nat({ori_path_list.name});

for f=0:Frame-1
    sign=get_randi();
    for row=1:floor(h/(patchsize-overlap))
        for col=1:floor(w/(patchsize-overlap))
            num_current=(col-1)*floor(h/(patchsize-overlap))+row+patch_each_frame*f;
            patch_dir=files_name{num_current};
            loc=file_path;
            inf_patch_name=strcat(loc,patch_dir);
            inf_patch=double(imread(inf_patch_name));
            inf_patch=inf_patch(:,:,1);
            
            
             %Only keep the structual information of infered patches  
            [~,str_pred,~] = get_patch_feature(inf_patch);
            ori_name=strcat( ori_patch_dir,'Ori',num2str(num_current),'.bmp');
            patch_ori=imread(ori_name);
            [L_ori,~,C_ori] = get_patch_feature(patch_ori);
            
            
            %Luminance adaptation and contrast visibility masking
            LA=get_LM(patch_ori);       
            if L_ori+sign*LA<0
                L_ori=0;
            elseif L_ori+sign*LA>255
                L_ori=255;
            else
                L_ori=L_ori+sign*LA;
            end
            %the f_C can be represented by a linear scaler
            C_ori=C_ori*0.9474;
            
            
            patch_new = Fea2Img(L_ori,C_ori,str_pred);
            patch=patch_new;

            %stich overlap patches together
            if col~=1&& row~=1   
                img_JND((row-1)*(patchsize-overlap)+1:(row-1)*(patchsize-overlap)+overlap,(col-1)*(patchsize-overlap)+1:(col-1)*(patchsize-overlap)+patchsize)= uint8((img_JND((row-1)*(patchsize-overlap)+1:(row-1)*(patchsize-overlap)+overlap,(col-1)*(patchsize-overlap)+1:(col-1)*(patchsize-overlap)+patchsize)+patch(1:overlap,1:patchsize))/2);                
                img_JND((row-1)*(patchsize-overlap)+(overlap+1):(row-1)*(patchsize-overlap)+patchsize,(col-1)*(patchsize-overlap)+1:(col-1)*(patchsize-overlap)+overlap)=uint8((img_JND((row-1)*(patchsize-overlap)+(overlap+1):(row-1)*(patchsize-overlap)+patchsize,(col-1)*(patchsize-overlap)+1:(col-1)*(patchsize-overlap)+overlap)+patch((overlap+1):patchsize,1:overlap))/2);              
                img_JND((row-1)*(patchsize-overlap)+(overlap+1):(row-1)*(patchsize-overlap)+patchsize,(col-1)*(patchsize-overlap)+(overlap+1):(col-1)*(patchsize-overlap)+patchsize)=patch((overlap+1):patchsize,(overlap+1):patchsize);
            elseif col==1&&row~=1               
                img_JND((row-1)*(patchsize-overlap)+1:(row-1)*(patchsize-overlap)+overlap,(col-1)*(patchsize-overlap)+1:(col-1)*(patchsize-overlap)+patchsize)= uint8((img_JND((row-1)*(patchsize-overlap)+1:(row-1)*(patchsize-overlap)+overlap,(col-1)*(patchsize-overlap)+1:(col-1)*(patchsize-overlap)+patchsize)+patch(1:overlap,1:patchsize))/2);           
                img_JND((row-1)*(patchsize-overlap)+(overlap+1):(row-1)*(patchsize-overlap)+patchsize,(col-1)*(patchsize-overlap)+1:(col-1)*(patchsize-overlap)+overlap)=patch((overlap+1):patchsize,1:overlap);                
                img_JND((row-1)*(patchsize-overlap)+(overlap+1):(row-1)*(patchsize-overlap)+patchsize,(col-1)*(patchsize-overlap)+(overlap+1):(col-1)*(patchsize-overlap)+patchsize)=patch((overlap+1):patchsize,(overlap+1):patchsize);
            elseif col~=1&&row==1
                img_JND((row-1)*(patchsize-overlap)+1:(row-1)*(patchsize-overlap)+patchsize,(col-1)*(patchsize-overlap)+1:(col-1)*(patchsize-overlap)+overlap)=uint8((img_JND((row-1)*(patchsize-overlap)+1:(row-1)*(patchsize-overlap)+patchsize,(col-1)*(patchsize-overlap)+1:(col-1)*(patchsize-overlap)+overlap)+patch(1:patchsize,1:overlap))/2);                
                img_JND((row-1)*(patchsize-overlap)+1:(row-1)*(patchsize-overlap)+patchsize,(col-1)*(patchsize-overlap)+(overlap+1):(col-1)*(patchsize-overlap)+patchsize)=patch(1:patchsize,(overlap+1):patchsize);
            elseif col==1&&row==1
                img_JND((row-1)*(patchsize-overlap)+1:(row-1)*(patchsize-overlap)+patchsize,(col-1)*(patchsize-overlap)+1:(col-1)*(patchsize-overlap)+patchsize)=patch;
            end         
        end
    end
    out_name=strcat('.\JND',num2str(f),'.bmp');
    imwrite(uint8(img_JND),out_name,'bmp');
    %ori_name=strcat(ori_img_dir,'Croped',num2str(f),'.png');
    ori_name=strcat(ori_img_dir,ori_files_name{f+1});
    img_ori=rgb2gray(imread(ori_name));
    img_ori=img_ori(1:floor(h/(patchsize-overlap))*(patchsize-overlap)+overlap,1:floor(w/(patchsize-overlap))*(patchsize-overlap)+overlap);
    img_jnd{f+1}=uint8(img_JND);
    stas_p(f+1)=psnr(img_ori,uint8(img_JND));
end
end

