function [img] = Fea2Img(miu,Con,Strc)
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
img=zeros(64,64);
img=img+miu;
img=Con*Strc+img;

end

