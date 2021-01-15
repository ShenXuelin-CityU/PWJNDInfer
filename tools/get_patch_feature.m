function [exp,V2,C2] = get_patch_feature(img)
img=double(img);
[m,n]=size(img);
exp=sum(sum(img))/(m*n);
C2=0;
for i=1:m
    for j=1:n
        C2=C2+(img(i,j)-exp)^2;
    end
end
C2=sqrt(C2);
 % C2=norm(img-miu,2);
 if C2~=0
     
  V2=img-exp;
  V2=V2/C2;
 else
     V2=zeros(m,n);
 end
  
end

