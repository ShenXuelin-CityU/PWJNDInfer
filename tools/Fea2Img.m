function [img] = Fea2Img(miu,Con,Strc)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
img=zeros(64,64);
img=img+miu;
img=Con*Strc+img;

end

